PRIORITY_SCORES = {
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1
}

def cri_priority_boost(cri):
    if cri is None:
        return 0.0
    elif cri <= 30:
        return 0.0
    elif cri <= 60:
        return 0.5
    elif cri <= 80:
        return 1.0
    else:
        return 1.5


def startup_security_advisor(context, cri=None):
    recommendations = []

    startup_size = context["startup_size"]
    infra = context["infrastructure"]
    systems = context["system_types"]
    data = context["data_sensitivity"]
    maturity = context["security_maturity"]

    # ---------- ACCESS CONTROL ----------

    if data == "high" or startup_size in ["medium", "large"]:
        recommendations.append({
            "control": "Multi-Factor Authentication (MFA)",
            "category": "Access Control",
            "priority": "Critical",
            "reason": "Sensitive data or growing team increases credential risk"
        })

    if startup_size != "small":
        recommendations.append({
            "control": "Role-Based Access Control (RBAC)",
            "category": "Access Control",
            "priority": "High",
            "reason": "Role separation is required as team size grows"
        })

    recommendations.append({
        "control": "Strong Password Policy",
        "category": "Access Control",
        "priority": "High",
        "reason": "Weak passwords are a leading cause of breaches"
    })

    # ---------- NETWORK SECURITY ----------

    if infra in ["cloud", "hybrid"]:
        recommendations.append({
            "control": "Firewall and Inbound Traffic Rules",
            "category": "Network Security",
            "priority": "Critical",
            "reason": "Externally exposed infrastructure requires traffic filtering"
        })

    if "database" in systems:
        recommendations.append({
            "control": "Network Segmentation",
            "category": "Network Security",
            "priority": "High",
            "reason": "Databases should not be directly exposed to application traffic"
        })

    if infra != "on-prem":
        recommendations.append({
            "control": "Secure Remote Access (VPN / Zero Trust)",
            "category": "Network Security",
            "priority": "Medium",
            "reason": "Remote access must be protected against unauthorized entry"
        })

    # ---------- DATA PROTECTION ----------

    if "database" in systems and data in ["medium", "high"]:
        recommendations.append({
            "control": "Encryption at Rest",
            "category": "Data Protection",
            "priority": "Critical",
            "reason": "Sensitive data must be protected if storage is compromised"
        })

    if infra in ["cloud", "hybrid"]:
        recommendations.append({
            "control": "Encryption in Transit (TLS)",
            "category": "Data Protection",
            "priority": "Critical",
            "reason": "Data in transit is vulnerable to interception"
        })

    # ---------- MONITORING ----------

    if maturity in ["none", "basic"]:
        recommendations.append({
            "control": "Centralized Logging",
            "category": "Monitoring",
            "priority": "High",
            "reason": "Without logs, incidents cannot be investigated"
        })

    recommendations.append({
        "control": "Security Monitoring and Alerts",
        "category": "Monitoring",
        "priority": "Medium",
        "reason": "Early detection reduces impact of attacks"
    })

    # ---------- RESILIENCE ----------

    recommendations.append({
        "control": "Regular Data Backups",
        "category": "Resilience",
        "priority": "Critical",
        "reason": "Backups protect against ransomware and data loss"
    })

    recommendations.append({
        "control": "Backup Restoration Testing",
        "category": "Resilience",
        "priority": "Medium",
        "reason": "Untested backups often fail during real incidents"
    })

    # ---------- PRIORITY SCORING & SORTING ----------

    boost = cri_priority_boost(cri)

    for rec in recommendations:
        base_score = PRIORITY_SCORES[rec["priority"]]
        rec["priority_score"] = base_score + boost

    recommendations = sorted(
        recommendations,
        key=lambda x: x["priority_score"],
        reverse=True
    )

    return recommendations
