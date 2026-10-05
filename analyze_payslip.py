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

file_path = "synthetic_payslip.pdf"

with open(file_path, "rb") as document:
    poller = client.begin_analyze_document(
        "prebuilt-layout",
        body=document,
    )

result = poller.result()

print("\n========== DOCUMENT ANALYSIS ==========\n")

print("Extracted text:")
print("----------------------------------------")

for page in result.pages:
    print(f"\n--- Page {page.page_number} ---")

    for line in page.lines:
        print(line.content)

print("\n========== TABLES ==========\n")

if result.tables:
    for table_index, table in enumerate(result.tables, start=1):
        print(f"--- Table {table_index} ---")

        for row in range(table.row_count):
            values = []

            for column in range(table.column_count):
                cell_text = ""

                for cell in table.cells:
                    if cell.row_index == row and cell.column_index == column:
                        cell_text = cell.content
                        break

                values.append(cell_text)

            print(" | ".join(values))
else:
    print("No tables detected.")