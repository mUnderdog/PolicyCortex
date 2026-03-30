def generate_security_roadmap(recommendations, cri=50):
    """
    Convert security recommendations into a phased implementation roadmap.
    """

    phase1 = []
    phase2 = []
    phase3 = []

    for rec in recommendations:

        control = rec["control"]
        priority = rec["priority"]

        if priority == "Critical":
            phase1.append(control)

        elif priority == "High":
            phase2.append(control)

        elif priority == "Medium":
            phase3.append(control)

    roadmap = {
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
        "cri": cri
    }

    return roadmap