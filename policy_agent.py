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

from rag_tool import search_finassist_knowledge


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
# 2. Define RAG / Policy Search Tool
# ---------------------------------------------------------

policy_tool = FunctionTool(
    name="search_finassist_knowledge",
    description=(
        "Search the FinAssist policy knowledge base using "
        "semantic retrieval and keyword-aware reranking. "
        "Use this tool whenever policy information is needed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The financial policy question or topic "
                    "that needs to be searched."
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
# 3. Create Policy Agent
# ---------------------------------------------------------

agent = project.agents.create_version(
    agent_name="finassist-policy-agent",
    definition=PromptAgentDefinition(
        model="gpt-5-mini",
        instructions="""
You are the FinAssist Policy Analysis Agent.

Your responsibility is to retrieve and explain
FinAssist financial policies using the approved
FinAssist knowledge base.

Rules:

1. Always use the search_finassist_knowledge tool
   when answering policy questions.

2. Use ONLY information returned by the
   FinAssist knowledge base.

3. Do not invent policy thresholds, rules,
   requirements, or decisions.

4. Identify the relevant policy ID and section
   whenever available.

5. Cite the source document and chunk when
   available.

6. Clearly distinguish policy information from
   financial calculations.

7. If the knowledge base does not contain the
   requested policy information, say so.

8. Do not make a final lending approval or
   rejection decision.

Your responsibility is policy interpretation,
not risk approval.
""",
        tools=[policy_tool],
    ),
)


print("\nPolicy Agent created successfully.")
print("Agent name:", agent.name)
print("Agent version:", agent.version)


# ---------------------------------------------------------
# 4. Create Conversation
# ---------------------------------------------------------

conversation = openai.conversations.create()


# ---------------------------------------------------------
# 5. Test Policy Agent
# ---------------------------------------------------------

question = (
    "My DTI is 27.59%. "
    "What does the FinAssist policy say about this DTI?"
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
        # Policy RAG Tool
        # -------------------------------------------------

        if item.name == "search_finassist_knowledge":

            result = search_finassist_knowledge(
                query=arguments["query"]
            )

            print(
                f"\nRAG returned "
                f"{len(result['results'])} results."
            )

            print("\nRetrieved policy evidence:")

            for result_item in result["results"]:

                print(
                    f"\nSource: "
                    f"{result_item['source']}"
                )

                print(
                    f"Policy ID: "
                    f"{result_item['policy_id']}"
                )

                print(
                    f"Section: "
                    f"{result_item['section']}"
                )

                print(
                    f"Chunk: "
                    f"{result_item['chunk_id']}"
                )

                print(
                    f"Similarity: "
                    f"{result_item['similarity']}"
                )

                print(
                    f"Text: "
                    f"{result_item['text']}"
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
    # No more tool calls
    # -----------------------------------------------------

    if not tool_calls_found:
        break


    # -----------------------------------------------------
    # Send tool result back to Policy Agent
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

print("\nPOLICY AGENT RESULT")
print("===================")

for item in current_response.output:

    if item.type == "message":

        for content in item.content:

            if hasattr(content, "text"):

                print(content.text)