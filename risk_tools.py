def assess_dti_risk(dti_ratio: float) -> dict:
    if dti_ratio < 0:
        raise ValueError("DTI cannot be negative.")

    if dti_ratio < 36:
        category = "preferred_range"
        review_requirement = "No additional DTI review triggered by this threshold"
        policy_section = "DTI BELOW 36%"

    elif dti_ratio <= 43:
        category = "additional_review"
        review_requirement = "Additional financial information review required"
        policy_section = "DTI FROM 36% TO 43%"

    else:
        category = "manual_risk_review"
        review_requirement = "Manual risk review required"
        policy_section = "DTI ABOVE 43%"

    return {
        "dti_ratio": dti_ratio,
        "category": category,
        "review_requirement": review_requirement,
        "policy_id": "DTI-001",
        "policy_section": policy_section,
        "final_approval_decision": "Not determined — human authorization required",
    }


if __name__ == "__main__":
    for dti in [27.59, 40.0, 48.0]:
        print(assess_dti_risk(dti))