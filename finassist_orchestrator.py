import re

from financial_tools import calculate_debt_to_income_ratio
from rag_query import retrieve, generate_answer


def extract_financial_values(question: str):
    """
    Extract simple income and debt values from the user question.

    This is intentionally simple for the first version.
    Later, the Document Intelligence Agent will provide
    structured financial values.
    """

    income_match = re.search(
        r"income\s*(?:is|=|of)?\s*[₹$]?\s*([\d,]+)",
        question,
        re.IGNORECASE,
    )

    debt_match = re.search(
        r"debt\s*(?:is|=|of)?\s*[₹$]?\s*([\d,]+)",
        question,
        re.IGNORECASE,
    )

    income = None
    debt = None

    if income_match:
        income = float(
            income_match.group(1).replace(",", "")
        )

    if debt_match:
        debt = float(
            debt_match.group(1).replace(",", "")
        )

    return income, debt


def run_finassist(question: str):

    print("\n==============================")
    print("FINASSIST SUPERVISOR")
    print("==============================")

    income, debt = extract_financial_values(question)

    # -----------------------------------------------------
    # Agent decision: calculation required
    # -----------------------------------------------------

    if income is not None and debt is not None:

        print("\n[Agent Decision]")
        print("Financial calculation detected.")

        print("\n[Tool]")
        print("calculate_debt_to_income_ratio")

        dti_result = calculate_debt_to_income_ratio(
            monthly_income=income,
            monthly_debt=debt,
        )

        print(
            f"DTI calculated: "
            f"{dti_result['dti_ratio']}%"
        )

        # -------------------------------------------------
        # Retrieve relevant policy
        # -------------------------------------------------

        print("\n[RAG]")
        print("Retrieving DTI policy...")

        policy_results = retrieve(
            "What does the policy say about this DTI?"
        )

        answer = generate_answer(
            f"""
The applicant has:

Monthly income: {income}
Monthly debt: {debt}
Calculated DTI: {dti_result['dti_ratio']}%

Explain what the FinAssist policy says about this DTI.
Do not make a lending approval decision.
""",
            policy_results,
        )

        print("\n==============================")
        print("FINASSIST RESULT")
        print("==============================")

        print(
            f"\nCalculated DTI: "
            f"{dti_result['dti_ratio']}%"
        )

        print("\nPolicy analysis:")
        print(answer)

        return

    # -----------------------------------------------------
    # Agent decision: policy question
    # -----------------------------------------------------

    print("\n[Agent Decision]")
    print("Policy/knowledge question detected.")

    print("\n[RAG]")
    print("Searching FinAssist knowledge base...")

    results = retrieve(question)

    answer = generate_answer(
        question,
        results,
    )

    print("\n==============================")
    print("FINASSIST RESULT")
    print("==============================")

    print(answer)


if __name__ == "__main__":

    question = input(
        "\nAsk FinAssist: "
    )

    run_finassist(question)