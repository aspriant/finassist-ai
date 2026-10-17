import json
import os

from dotenv import load_dotenv
from azure.identity import AzureCliCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, FunctionTool, Tool

from financial_tools import calculate_debt_to_income_ratio
from rag_tool import search_finassist_knowledge
from document_tools import analyze_payslip

from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseInputParam,
)
from risk_tools import assess_dti_risk


load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
AGENT_NAME = os.environ["FOUNDRY_AGENT_NAME"]


# ---------------------------------------------------------
# 1. Connect to Microsoft Foundry
# ---------------------------------------------------------

project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=AzureCliCredential(),
)

openai = project.get_openai_client()


# ---------------------------------------------------------
# 2. Define DTI tool
# ---------------------------------------------------------

dti_tool = FunctionTool(
    name="calculate_debt_to_income_ratio",
    description=(
        "Calculate a customer's monthly debt-to-income ratio. "
        "Use this tool whenever the user asks for a DTI calculation "
        "or provides monthly income and monthly debt that need to be "
        "converted into a DTI percentage."
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


# ---------------------------------------------------------
# 3. Define RAG knowledge search tool
# ---------------------------------------------------------

rag_tool = FunctionTool(
    name="search_finassist_knowledge",
    description=(
        "Search the FinAssist policy knowledge base. "
        "Use this tool for questions about DTI policy, lending policy, "
        "income verification, document requirements, financial policy "
        "rules, thresholds, or other information that should be "
        "grounded in FinAssist knowledge documents."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The policy or knowledge question to search "
                    "in the FinAssist knowledge base."
                ),
            },
        },
        "required": [
            "query",
        ],
        "additionalProperties": False,
    },
    strict=True,
)

# ---------------------------------------------------------
# 4. Define payslip analysis tool
# ---------------------------------------------------------

payslip_tool = FunctionTool(
    name="analyze_payslip",
    description=(
        "Analyze the payslip using Azure AI Document Intelligence "
        "and return structured financial information including "
        "gross monthly income, deductions, net salary, earnings, "
        "employee information and pay period."
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
# Define deterministic risk assessment tool
# ---------------------------------------------------------

risk_tool = FunctionTool(
    name="assess_dti_risk",
    description=(
        "Classify a calculated DTI using FinAssist policy DTI-001. "
        "Use this after the DTI calculation is available. "
        "Returns the applicable category and review requirement. "
        "Does not make a final lending decision."
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
# 4. Register both tools
# ---------------------------------------------------------

tools: list[Tool] = [
    dti_tool,
    rag_tool,
    payslip_tool,
    risk_tool,
]


# ---------------------------------------------------------
# 5. Create new FinAssist agent version
# ---------------------------------------------------------

agent = project.agents.create_version(
    agent_name=AGENT_NAME,
    definition=PromptAgentDefinition(
        model="gpt-5-mini",
        instructions="""
You are FinAssist Supervisor, an enterprise financial document
analysis AI agent.

Your responsibility is to coordinate financial analysis using
deterministic financial tools and the FinAssist policy knowledge base.

TOOL RULES

1. Use calculate_debt_to_income_ratio whenever a DTI calculation
   is requested.

2. Never perform DTI arithmetic yourself when the DTI tool is available.

3. Use search_finassist_knowledge for policy questions.

4. If the user asks for BOTH a financial calculation and policy
   interpretation, use BOTH tools.

5. After receiving a DTI calculation, use the policy knowledge
   tool when the user asks what the policy says about the result.

6. Do not invent financial values, policies, thresholds, or
   customer information.

7. Clearly distinguish calculated values from policy information.

8. If required information is missing, ask the user for it.

9. Policy answers must be grounded in the FinAssist knowledge base
   whenever relevant.

10. Do not make final financial approval or rejection decisions.

11. Cases requiring authorization, exception handling, or final
    approval must be escalated to an appropriate human reviewer.

12. Protect sensitive financial information.

13. Use assess_dti_risk after receiving a valid DTI calculation
    when a risk classification is requested.

14. Pass the exact dti_ratio returned by the deterministic DTI tool
    to assess_dti_risk. Do not recalculate or invent the value.

15. Explain the risk category and review requirement returned
    by the risk tool. Do not make a final lending decision.

RESPONSE RULES

When presenting a financial calculation:

- Clearly state the calculated result.
- Explain the calculation briefly.
- Identify the policy evidence when policy information is used.
- Mention the source document when available.

For combined calculation and policy questions, structure the
response as:

1. Calculation
2. Policy interpretation
3. Important assumptions or missing verification
4. Human-review requirement when applicable
""",
        tools=tools,
    ),
)


print("Agent version created successfully.")
print(f"Agent name: {agent.name}")
print(f"Agent version: {agent.version}")


# ---------------------------------------------------------
# 6. Create conversation
# ---------------------------------------------------------

conversation = openai.conversations.create()


# ---------------------------------------------------------
# 7. Test combined agentic workflow
# ---------------------------------------------------------

question = (
    "Analyze my payslip. "
    "Use the gross monthly income extracted from the payslip "
    "and calculate my DTI using monthly debt obligations of 40000. "
    "Retrieve the relevant FinAssist policy evidence, "
    "classify the DTI using the risk tool, and provide a consolidated "
    "report with assumptions and human-review requirements."
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
# 8. Process tool calls until agent returns final response
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

        # ---------------------------------------------
        # DTI tool
        # ---------------------------------------------

        if item.name == "calculate_debt_to_income_ratio":

            result = calculate_debt_to_income_ratio(
                monthly_income=arguments["monthly_income"],
                monthly_debt=arguments["monthly_debt"],
            )

            print(f"Python DTI result: {result}")

            input_list.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=item.call_id,
                    output=json.dumps(result),
                )
            )

        # ---------------------------------------------
        # RAG tool
        # ---------------------------------------------

        elif item.name == "search_finassist_knowledge":

            result = search_finassist_knowledge(
                query=arguments["query"]
            )

            print(
                f"RAG returned {len(result['results'])} results."
            )

            input_list.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=item.call_id,
                    output=json.dumps(result),
                )
            )

        # ---------------------------------------------
        # Payslip analysis tool
        # ---------------------------------------------

        elif item.name == "analyze_payslip":

            result = analyze_payslip()

            print("Payslip analysis completed.")

            print(
                f"Gross monthly income: "
                f"{result['gross_monthly_income']}"
            )

            input_list.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=item.call_id,
                    output=json.dumps(result),
                )
            )
        # -------------------------------------------------
        # Risk assessment tool
        # -------------------------------------------------

        elif item.name == "assess_dti_risk":

            result = assess_dti_risk(
                dti_ratio=arguments["dti_ratio"]
            )

            print(f"Risk assessment result: {result}")

            input_list.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=item.call_id,
                    output=json.dumps(result),
                )
            )

        else:

            print(f"Unknown tool requested: {item.name}")

    # -------------------------------------------------
    # No more tools = final answer
    # -------------------------------------------------

    if not tool_calls_found:

        print("\n--------------------------------")
        print("FINAL FINASSIST RESPONSE")
        print("--------------------------------")
        print(current_response.output_text)

        break

    # -------------------------------------------------
    # Send tool results back to agent
    # -------------------------------------------------

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

