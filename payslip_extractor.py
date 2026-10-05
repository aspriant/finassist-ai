import os
import json

from dotenv import load_dotenv
from azure.identity import AzureCliCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient


load_dotenv()

ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
DOCUMENT_PATH = "synthetic_payslip.pdf"


def analyze_document():
    if not ENDPOINT:
        raise ValueError(
            "DOCUMENT_INTELLIGENCE_ENDPOINT is not configured."
        )

    if not os.path.exists(DOCUMENT_PATH):
        raise FileNotFoundError(
            f"Document not found: {DOCUMENT_PATH}"
        )

    credential = AzureCliCredential()

    client = DocumentIntelligenceClient(
        endpoint=ENDPOINT,
        credential=credential,
    )

    with open(DOCUMENT_PATH, "rb") as document:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            body=document,
        )

    return poller.result()


def extract_tables(result):
    tables = []

    for table in result.tables:
        rows = []

        for row_index in range(table.row_count):
            row = []

            for column_index in range(table.column_count):
                value = ""

                for cell in table.cells:
                    if (
                        cell.row_index == row_index
                        and cell.column_index == column_index
                    ):
                        value = cell.content
                        break

                row.append(value)

            rows.append(row)

        tables.append(rows)

    return tables


def find_value(rows, label):
    for row in rows:
        if len(row) >= 2:
            if row[0].strip().lower() == label.lower():
                return row[1].strip()

    return None


def parse_amount(value):
    if not value:
        return None

    value = (
        value.replace("₹", "")
        .replace(",", "")
        .replace("INR", "")
        .strip()
    )

    try:
        return float(value)
    except ValueError:
        return None


def build_payslip_data(tables):
    employee_table = tables[0] if len(tables) > 0 else []
    earnings_table = tables[1] if len(tables) > 1 else []
    deductions_table = tables[2] if len(tables) > 2 else []
    summary_table = tables[3] if len(tables) > 3 else []

    payslip = {
        "document_type": "payslip",

        "employee": {
            "name": find_value(employee_table, "Employee Name"),
            "employee_id": find_value(employee_table, "Employee ID"),
            "employer": find_value(employee_table, "Employer"),
            "designation": find_value(employee_table, "Designation"),
        },

        "pay_period": find_value(
            employee_table,
            "Pay Period"
        ),

        "pay_date": find_value(
            employee_table,
            "Pay Date"
        ),

        "gross_monthly_income": parse_amount(
            find_value(summary_table, "Gross Salary")
        ),

        "total_deductions": parse_amount(
            find_value(summary_table, "Total Deductions")
        ),

        "net_salary": parse_amount(
            find_value(summary_table, "Net Salary")
        ),

        "earnings": {
            "basic_salary": parse_amount(
                find_value(earnings_table, "Basic Salary")
            ),
            "house_rent_allowance": parse_amount(
                find_value(
                    earnings_table,
                    "House Rent Allowance"
                )
            ),
            "special_allowance": parse_amount(
                find_value(
                    earnings_table,
                    "Special Allowance"
                )
            ),
            "transport_allowance": parse_amount(
                find_value(
                    earnings_table,
                    "Transport Allowance"
                )
            ),
        },

        "deductions": {
            "provident_fund": parse_amount(
                find_value(
                    deductions_table,
                    "Provident Fund"
                )
            ),
            "professional_tax": parse_amount(
                find_value(
                    deductions_table,
                    "Professional Tax"
                )
            ),
            "income_tax": parse_amount(
                find_value(
                    deductions_table,
                    "Income Tax / TDS"
                )
            ),
            "other_deductions": parse_amount(
                find_value(
                    deductions_table,
                    "Other Deductions"
                )
            ),
        },
    }

    return payslip


def main():
    print("\n========== FINASSIST PAYSLIP EXTRACTION ==========\n")

    result = analyze_document()

    tables = extract_tables(result)

    payslip = build_payslip_data(tables)

    print(
        json.dumps(
            payslip,
            indent=2
        )
    )


if __name__ == "__main__":
    main()