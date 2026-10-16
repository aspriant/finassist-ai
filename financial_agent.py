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

from financial_tools import calculate_debt_to_income_ratio


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
# 2. Define DTI Calculation Tool
# ---------------------------------------------------------

dti_tool = FunctionTool(
    name="calculate_debt_to_income_ratio",
    description=(
        "Calculate the monthly Debt-to-Income Ratio (DTI). "
        "Use verified gross monthly income and total "
        "monthly debt obligations. "
        "This tool performs the calculation deterministically."
    ),
    parameters={
        "type": "object",
        "properties": {
            "monthly_income": {
                "type": "number",
                "description": (
                    "Verified gross monthly income."
                ),
            },
            "monthly_debt": {
                "type": "number",
                "description": (
                    "Total verified recurring monthly "
                    "debt obligations."
                ),
            },
        },
        "required": [
            "monthly_income",
            "monthly_debt",
        ],
        "additionalProperties": False,
    },
    strict=True,
)


# ---------------------------------------------------------
# 3. Create Financial Agent
# ---------------------------------------------------------

agent = project.agents.create_version(
    agent_name="finassist-financial-agent",
    definition=PromptAgentDefinition(
        model="gpt-5-mini",
        instructions="""
You are the FinAssist Financial Analysis Agent.

Your responsibility is to perform deterministic financial
calculations using approved financial tools.

For DTI calculations:

1. Use the calculate_debt_to_income_ratio tool.
2. Never perform financial arithmetic mentally.
3. Use the exact income and debt values provided.
4. Do not invent missing financial values.
5. Clearly identify the values used in the calculation.
6. Return the calculated DTI percentage.
7. Clearly state if required information is missing.

You are NOT responsible for:

- extracting information from documents
- interpreting lending policies
- making risk decisions
- approving or rejecting applications

Those responsibilities belong to other FinAssist agents.
""",
        tools=[dti_tool],
    ),
)


print("\nFinancial Agent created successfully.")
print("Agent name:", agent.name)
print("Agent version:", agent.version)


# ---------------------------------------------------------
# 4. Create Conversation
# ---------------------------------------------------------

conversation = openai.conversations.create()


# ---------------------------------------------------------
# 5. Test Financial Agent
# ---------------------------------------------------------

question = (
    "Calculate my DTI using a gross monthly income "
    "of 145000 and monthly debt obligations of 40000."
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
# 6. Process Tool Calls
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
        # DTI Calculation
        # -------------------------------------------------

        if item.name == "calculate_debt_to_income_ratio":

            result = calculate_debt_to_income_ratio(
                monthly_income=arguments["monthly_income"],
                monthly_debt=arguments["monthly_debt"],
            )

            print("\nPython DTI result:")
            print(result)

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
    # No more tool calls
    # -----------------------------------------------------

    if not tool_calls_found:
        break


    # -----------------------------------------------------
    # Send tool result back to Financial Agent
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
# 7. Display Final Response
# ---------------------------------------------------------

print("\nFINANCIAL AGENT RESULT")
print("======================")

for item in current_response.output:

    if item.type == "message":

        for content in item.content:

            if hasattr(content, "text"):

                print(content.text)