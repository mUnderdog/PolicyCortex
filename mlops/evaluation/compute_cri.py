import pandas as pd
import joblib
from pathlib import Path
import xgboost as xgb

from models.risk_model.cri import compute_cri, get_severity_weight

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path(r"D:\Disk D\Study Material\AI & ML\PolicyCortex\data\processed\cicids2017_v1.csv")
MODEL_PATH = Path(r"D:\Disk D\Study Material\AI & ML\PolicyCortex\models\risk_model\xgb_risk_model.pkl")
OUTPUT_PATH = Path(r"D:\Disk D\Study Material\AI & ML\PolicyCortex\data\processed\cicids2017_with_cri.csv")

# -----------------------------
# Load Data & Model
# -----------------------------
print("[INFO] Loading data and model...")
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

X = df.drop(columns=["target"])
y = df["target"]

# -----------------------------
# Predict Probabilities
# -----------------------------
print("[INFO] Predicting probabilities...")
probs = model.predict_proba(X)[:, 1]
df["attack_probability"] = probs

# -----------------------------
# Assign Severity
# -----------------------------
# If Label column is gone, simulate severity based on probability
def severity_proxy(p):
    if p > 0.9:
        return 1.0
    elif p > 0.7:
        return 0.8
    elif p > 0.5:
        return 0.6
    else:
        return 0.2

df["severity_weight"] = df["attack_probability"].apply(severity_proxy)

# -----------------------------
# Compute CRI
# -----------------------------
df["cyber_risk_index"] = df.apply(
    lambda row: compute_cri(
        row["attack_probability"],
        row["severity_weight"]
    ),
    axis=1
)

print("[INFO] CRI stats:")
print(df["cyber_risk_index"].describe())

# -----------------------------
# Policy Band Mapping
# -----------------------------
def map_policy_band(cri):
    if cri <= 10:
        return "Low", "Monitor"
    elif cri <= 30:
        return "Moderate", "Enforce baseline security controls"
    elif cri <= 60:
        return "High", "Escalate to SOC for investigation"
    elif cri <= 80:
        return "Severe", "Activate incident response"
    else:
        return "Critical", "Emergency response and policy enforcement"


df[["risk_level", "recommended_action"]] = df["cyber_risk_index"].apply(
    lambda x: pd.Series(map_policy_band(x))
)


# -----------------------------
# Save
# -----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"[INFO] CRI dataset saved to {OUTPUT_PATH}")
