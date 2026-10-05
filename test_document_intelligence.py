import os

from dotenv import load_dotenv
from azure.identity import AzureCliCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient


load_dotenv()

endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")

if not endpoint:
    raise ValueError("DOCUMENT_INTELLIGENCE_ENDPOINT is not configured.")

credential = AzureCliCredential()

client = DocumentIntelligenceClient(
    endpoint=endpoint,
    credential=credential,
)

print("Document Intelligence client initialized successfully.")
print(f"Endpoint: {endpoint}")