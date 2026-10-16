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

from risk_tools import assess_dti_risk

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
# 2. Define deterministic risk assessment tool
# ---------------------------------------------------------

risk_tool = FunctionTool(
    name="assess_dti_risk",
    description=(
        "Classify a calculated DTI using the FinAssist DTI-001 "
        "policy thresholds. Use this tool for DTI risk classification. "
        "It does not make a lending approval decision."
    ),
    parameters={
        "type": "object",
        "properties": {
            "dti_ratio": {
                "type": "number",
                "description": "Calculated DTI percentage, such as 27.59.",
            },
        },
        "required": ["dti_ratio"],
        "additionalProperties": False,
    },
    strict=True,
)

# ---------------------------------------------------------
# 3. Create Risk Agent
# ---------------------------------------------------------

agent = project.agents.create_version(
    agent_name="finassist-risk-agent",
    definition=PromptAgentDefinition(
        model="gpt-5-mini",
        instructions="""
You are the FinAssist Risk Classification Agent.

Your responsibility is to classify a supplied DTI using
the approved assess_dti_risk tool.

Rules:
1. Use the tool for DTI classification.
2. Do not invent financial values or policy thresholds.
3. Report the category and review requirement returned by the tool.
4. Identify policy ID and section when provided.
5. Never approve or reject a loan.
6. State that final authorization remains with a human reviewer.
7. Do not claim that a complete credit risk assessment was performed
   when only DTI was assessed.

You classify DTI policy categories; you do not make final decisions.
""",
        tools=[risk_tool],
    ),
)

print("\nRisk Agent created successfully.")
print("Agent name:", agent.name)
print("Agent version:", agent.version)

# ---------------------------------------------------------
# 4. Test the Risk Agent
# ---------------------------------------------------------

conversation = openai.conversations.create()

question = (
    "Classify a DTI of 27.59% using the FinAssist DTI policy. "
    "Tell me the category, applicable policy section, and review requirement."
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
# 5. Process function calls
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

        if item.name == "assess_dti_risk":
            result = assess_dti_risk(
                dti_ratio=arguments["dti_ratio"]
            )

            print("\nDeterministic risk result:")
            print(json.dumps(result, indent=2))

            input_list.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=item.call_id,
                    output=json.dumps(result),
                )
            )
        else:
            raise RuntimeError(
                f"Unexpected tool requested: {item.name}"
            )

    if not tool_calls_found:
        break

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
# 6. Display final response
# ---------------------------------------------------------

print("\nRISK AGENT RESULT")
print("=================")

for item in current_response.output:
    if item.type == "message":
        for content in item.content:
            if hasattr(content, "text"):
                print(content.text)