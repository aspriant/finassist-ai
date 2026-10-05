from payslip_extractor import analyze_document, extract_tables, build_payslip_data
from financial_tools import calculate_debt_to_income_ratio


def main():
    print("\n========== FINASSIST PAYSLIP DTI ANALYSIS ==========\n")

    # 1. Analyze payslip
    result = analyze_document()

    # 2. Extract tables
    tables = extract_tables(result)

    # 3. Convert to structured financial data
    payslip = build_payslip_data(tables)

    income = payslip["gross_monthly_income"]

    # Synthetic debt value for testing
    monthly_debt = 40000

    # 4. Deterministic DTI calculation
    dti_result = calculate_debt_to_income_ratio(
        monthly_income=income,
        monthly_debt=monthly_debt,
    )

    print("Employee:", payslip["employee"]["name"])
    print("Employer:", payslip["employee"]["employer"])
    print("Pay Period:", payslip["pay_period"])

    print("\nFinancial Information")
    print("---------------------")
    print(f"Gross Monthly Income: ₹{income:,.2f}")
    print(f"Monthly Debt:         ₹{monthly_debt:,.2f}")

    print("\nDTI Result")
    print("----------")
    print(f"DTI: {dti_result['dti_ratio']:.2f}%")

    print("\nCalculation completed using deterministic financial tool.")


if __name__ == "__main__":
    main()