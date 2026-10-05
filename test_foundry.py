# Imports Python's operating-system helpers so environment variables can be read.
import os

# Imports the helper that reads key-value pairs from a local .env file.
from dotenv import load_dotenv
# Imports Azure's credential chain, which discovers available local or hosted credentials.
from azure.identity import DefaultAzureCredential
# Imports the client used to connect to an Azure AI Foundry project.
from azure.ai.projects import AIProjectClient

# Loads variables from a .env file into the process environment when that file exists.
load_dotenv()

# Reads the Azure AI Foundry project endpoint from the environment.
PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
# Reads the name of the Foundry agent to use from the environment.
AGENT_NAME = os.environ["FOUNDRY_AGENT_NAME"]

# Shows that the script is about to establish the project connection.
print("Connecting to Microsoft Foundry...")

# Creates a credential that can use available Azure CLI, managed identity, or developer credentials.
credential = DefaultAzureCredential()

# Builds a Foundry project client using the endpoint and discovered credentials.
project = AIProjectClient(
    # Supplies the URL of the Foundry project.
    endpoint=PROJECT_ENDPOINT,
    # Supplies the credential used to authenticate requests.
    credential=credential,
)

# Confirms that the project client was created successfully.
print("Connected to Foundry project.")

# Creates an OpenAI-compatible client configured to communicate with the selected agent.
openai = project.get_openai_client(
    # Identifies which Foundry agent should receive requests.
    agent_name=AGENT_NAME
)

# Displays the agent selected by the environment configuration.
print(f"Connected to agent: {AGENT_NAME}")

# Creates a new server-side conversation so requests can share conversation history.
conversation = openai.conversations.create()

# Sends a prompt to the selected agent within the newly created conversation.
response = openai.responses.create(
    # Associates this request with the conversation created above.
    conversation=conversation.id,
    # Provides the user's question for the agent to answer.
    input="What is FinAssist AI?",
)

# Prints a simple heading before showing the agent's answer.
print("\n--- FinAssist Response ---")
# Prints the text returned by the agent from the response object.
print(response.output_text)