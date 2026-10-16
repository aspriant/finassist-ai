import json
import os

from dotenv import load_dotenv

from azure.identity import AzureCliCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    FunctionTool,
)

from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseInputParam,
)

from document_tools import analyze_payslip


load_dotenv()


# ---------------------------------------------------------
# 1. Connect to Microsoft Foundry
# ---------------------------------------------------------

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]

project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=AzureCliCredential(),
)

openai = project.get_openai_client()


# ---------------------------------------------------------
# 2. Define Payslip Analysis Tool
# ---------------------------------------------------------

payslip_tool = FunctionTool(
    name="analyze_payslip",
    description=(
        "Analyze a payslip using Azure AI Document Intelligence "
        "and return structured employee and salary information."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    strict=True,
)


# ---------------------------------------------------------
# 4. Create Document Agent
# ---------------------------------------------------------

agent = project.agents.create_version(
    agent_name="finassist-document-agent",
    definition=PromptAgentDefinition(
        model="gpt-5-mini",
        instructions="""
You are the FinAssist Document Analysis Agent.

Your responsibility is to extract structured financial
information from financial documents.

For payslips:

1. Use the analyze_payslip tool.
2. Do not manually calculate financial values.
3. Do not invent missing information.
4. Return only information actually extracted from the document.
5. Clearly identify missing information.
6. Preserve the original financial values.
7. Return structured information that another agent can use.

You are NOT responsible for:

- calculating DTI
- interpreting lending policies
- making risk decisions
- approving or rejecting applications

Those responsibilities belong to other FinAssist agents.
""",
        tools=[payslip_tool],
    ),
)


print("\nDocument Agent created successfully.")
print("Agent name:", agent.name)
print("Agent version:", agent.version)


# ---------------------------------------------------------
# 5. Create Conversation
# ---------------------------------------------------------

conversation = openai.conversations.create()


# ---------------------------------------------------------
# 6. Test Document Agent
# ---------------------------------------------------------

question = (
    "Analyze my payslip and extract the employee "
    "and salary information."
)

print("\nUSER QUESTION")
print("=============")
print(question)


response = openai.responses.create(
    input=question,
    conversation=conversation.id,
    extra_body={
        "agent_reference": {
            "name": agent.name,
            "type": "agent_reference",
        }
    },
)


# ---------------------------------------------------------
# 7. Process Tool Calls
# ---------------------------------------------------------

current_response = response


while True:

    input_list: ResponseInputParam = []

    tool_calls_found = False

    for item in current_response.output:

        if item.type != "function_call":
            continue

        tool_calls_found = True

        print(f"\nTool requested: {item.name}")
        print(f"Arguments: {item.arguments}")

        arguments = json.loads(item.arguments)

        # -------------------------------------------------
        # Payslip Analysis Tool
        # -------------------------------------------------

        if item.name == "analyze_payslip":

            result = analyze_payslip()

            print("\nPayslip analysis completed.")

            print(
                f"Gross monthly income: "
                f"{result['gross_monthly_income']}"
            )

            print("\nExtracted payslip data:")
            print(
                json.dumps(
                    result,
                    indent=2,
                )
            )

            input_list.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=item.call_id,
                    output=json.dumps(result),
                )
            )

        else:

            print(
                f"Unknown tool requested: {item.name}"
            )


    # -----------------------------------------------------
    # No tool calls = final answer
    # -----------------------------------------------------

    if not tool_calls_found:

        break


    # -----------------------------------------------------
    # Send tool results back to the agent
    # -----------------------------------------------------

    current_response = openai.responses.create(
        input=input_list,
        conversation=conversation.id,
        extra_body={
            "agent_reference": {
                "name": agent.name,
                "type": "agent_reference",
            }
        },
    )


# ---------------------------------------------------------
# 8. Display Final Response
# ---------------------------------------------------------

print("\nDOCUMENT AGENT RESULT")
print("=====================")

for item in current_response.output:

    if item.type == "message":

        for content in item.content:

            if hasattr(content, "text"):

                print(content.text)