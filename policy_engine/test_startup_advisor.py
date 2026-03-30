from policy_engine.startup_advisor import startup_security_advisor

# context_early = {
#     "startup_size": "small",
#     "infrastructure": "cloud",
#     "system_types": ["web", "database"],
#     "data_sensitivity": "high",
#     "security_maturity": "basic"
# }

# print("\n=== Early-Stage Startup (No CRI) ===")
# recs = startup_security_advisor(context_early)

# for r in recs:
#     print(f"{r['priority']} ({r['priority_score']}): {r['control']}")


# print("\n=== Early-Stage Startup (With CRI = 85) ===")
# recs_cri = startup_security_advisor(context_early, cri=85)

# for r in recs_cri:
#     print(f"{r['priority']} ({r['priority_score']}): {r['control']}")


context_mature = {
    "startup_size": "medium",
    "infrastructure": "hybrid",
    "system_types": ["web", "api"],
    "data_sensitivity": "medium",
    "security_maturity": "moderate"
}

print("\n=== Mature Startup (CRI = 40) ===")
recs_mature = startup_security_advisor(context_mature, cri=40)

for r in recs_mature:
    print(f"{r['priority']} ({r['priority_score']}): {r['control']}")
