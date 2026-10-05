from payslip_extractor import (
    analyze_document,
    extract_tables,
    build_payslip_data,
)


def analyze_payslip() -> dict:
    """
    Analyze the synthetic payslip using Azure AI Document Intelligence
    and return normalized financial information.
    """

    result = analyze_document()

    tables = extract_tables(result)

    payslip = build_payslip_data(tables)

    return payslip


if __name__ == "__main__":
    result = analyze_payslip()

    print("\n========== PAYSLIP TOOL RESULT ==========\n")

    import json

    print(json.dumps(result, indent=2))