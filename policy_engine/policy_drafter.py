def draft_policy_template(control, priority, context):
    return f"""
Policy Title: {control}

Policy Statement:
All systems and users within the organization must comply with this policy to reduce cybersecurity risk.

Scope:
This policy applies to all systems, applications, and personnel relevant to the control.

Enforcement:
Non-compliance may result in restricted access or corrective actions based on organizational guidelines.

Priority Level:
{priority}
""".strip()
