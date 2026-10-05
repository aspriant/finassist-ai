# Defines a reusable function for calculating a borrower's debt-to-income ratio.
def calculate_debt_to_income_ratio(
    # Accepts the borrower's total monthly gross income as a decimal number.
    monthly_income: float,
    # Accepts the borrower's total monthly debt obligations as a decimal number.
    monthly_debt: float,
    # Declares that the function returns a dictionary containing the calculation results.
) -> dict:
    # Documents the purpose, formula, inputs, and output of this financial tool.
    """
    Calculate the monthly debt-to-income ratio.

    DTI = (monthly debt / monthly gross income) * 100

    Args:
        monthly_income: Monthly gross income.
        monthly_debt: Total monthly debt obligations.

    Returns:
        A dictionary containing the input values and calculated DTI percentage.
    """

    # Rejects zero or negative income because it cannot be used as a valid divisor.
    if monthly_income <= 0:
        # Raises a clear error so callers know why the calculation cannot continue.
        raise ValueError("Monthly income must be greater than zero.")

    # Rejects negative debt because debt obligations cannot be below zero.
    if monthly_debt < 0:
        # Raises a clear error so callers can correct the invalid debt value.
        raise ValueError("Monthly debt cannot be negative.")

    # Divides monthly debt by monthly income and converts the ratio to a percentage.
    dti_ratio = (monthly_debt / monthly_income) * 100

    # Returns the original inputs together with the rounded DTI result.
    return {
        # Includes the monthly income used in the calculation.
        "monthly_income": monthly_income,
        # Includes the monthly debt used in the calculation.
        "monthly_debt": monthly_debt,
        # Rounds the percentage to two decimal places for readable financial output.
        "dti_ratio": round(dti_ratio, 2),
    }