import json
import os

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, FunctionTool, Tool

from financial_tools import calculate_debt_to_income_ratio
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseInputParam,
)

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
AGENT_NAME = os.environ["FOUNDRY_AGENT_NAME"]


# ---------------------------------------------------------
# 1. Connect to Microsoft Foundry
# ---------------------------------------------------------

project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
)

openai = project.get_openai_client()


# ---------------------------------------------------------
# 2. Define the DTI function as an AI tool
# ---------------------------------------------------------

dti_tool = FunctionTool(
    name="calculate_debt_to_income_ratio",
    description=(
        "Calculate a customer's monthly debt-to-income ratio. "
        "Use this tool whenever the user asks for a DTI calculation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "monthly_income": {
                "type": "number",
                "description": "Customer's monthly gross income.",
            },
            "monthly_debt": {
                "type": "number",
                "description": "Customer's total monthly debt obligations.",
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

tools: list[Tool] = [dti_tool]


# ---------------------------------------------------------
# 3. Create a new version of FinAssist with the tool
# ---------------------------------------------------------

agent = project.agents.create_version(
    agent_name=AGENT_NAME,
    definition=PromptAgentDefinition(
        model="gpt-5-mini",
        instructions="""
You are FinAssist Supervisor, an enterprise financial document
analysis AI agent.

You coordinate financial analysis workflows.

Rules:
1. Never invent financial values.
2. Use the calculate_debt_to_income_ratio tool whenever a
   DTI calculation is requested.
3. Do not perform DTI arithmetic yourself when the tool is available.
4. Clearly explain the calculation result.
5. Do not make final financial approval decisions.
6. If required information is missing, ask for it.
7. Protect sensitive financial information.
""",
        tools=tools,
    ),
)

print("Agent version created successfully.")
print(f"Agent name: {agent.name}")
print(f"Agent version: {agent.version}")


# ---------------------------------------------------------
# 4. Create a conversation
# ---------------------------------------------------------

conversation = openai.conversations.create()


# ---------------------------------------------------------
# 5. Ask the agent to calculate DTI
# ---------------------------------------------------------

response = openai.responses.create(
    input=(
        "A customer has a monthly gross income of 150000 "
        "and monthly debt obligations of 40000. "
        "Calculate the customer's debt-to-income ratio."
    ),
    conversation=conversation.id,
    extra_body={
        "agent_reference": {
            "name": agent.name,
            "type": "agent_reference",
        }
    },
)


# ---------------------------------------------------------
# 6. Process tool calls
# ---------------------------------------------------------

input_list: ResponseInputParam = []

for item in response.output:

    if item.type == "function_call":

        print(f"\nTool requested: {item.name}")
        print(f"Arguments: {item.arguments}")

        if item.name == "calculate_debt_to_income_ratio":

            arguments = json.loads(item.arguments)

            result = calculate_debt_to_income_ratio(
                monthly_income=arguments["monthly_income"],
                monthly_debt=arguments["monthly_debt"],
            )

            print(f"Python tool result: {result}")

            input_list.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=item.call_id,
                    output=json.dumps(result),
                )
            )


# ---------------------------------------------------------
# 7. Send the tool result back to the agent
# ---------------------------------------------------------

if input_list:

    final_response = openai.responses.create(
        input=input_list,
        conversation=conversation.id,
        extra_body={
            "agent_reference": {
                "name": agent.name,
                "type": "agent_reference",
            }
        },
    )

    print("\n--------------------------------")
    print("FINAL FINASSIST RESPONSE")
    print("--------------------------------")
    print(final_response.output_text)

else:

    print("\nAgent did not request the DTI tool.")
    print(response.output_text)