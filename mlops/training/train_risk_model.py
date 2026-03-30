import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import mlflow
import mlflow.xgboost

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("data/processed/cicids2017_v1.csv")
MODEL_PATH = Path("models/risk_model/xgb_risk_model.pkl")

# -----------------------------
# Load Data
# -----------------------------
print("[INFO] Loading processed dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["target"])
y = df["target"]

print("[INFO] Dataset shape:", X.shape)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.array([0, 1])
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight_dict = {
    0: class_weights[0],
    1: class_weights[1]
}

print("Class Weights:", class_weight_dict)

# -----------------------------
# GPU XGBoost Model
# -----------------------------
print("[INFO] Initializing GPU XGBoost...")

model = xgb.XGBClassifier(
    tree_method="hist",     # use histogram method
    device="cuda",          # tell XGBoost to use GPU
    max_depth=6,
    n_estimators=200,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

# -----------------------------
# MLflow Tracking
# -----------------------------
mlflow.set_experiment("Cyber_Risk_Model_GPU")

with mlflow.start_run():

    print("[INFO] Training model...")
    sample_weights = y_train.map(class_weight_dict)

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    from sklearn.metrics import confusion_matrix

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", auc)

    # Log metrics
    mlflow.log_metric("roc_auc", auc)

    # Log model
    mlflow.xgboost.log_model(model, "xgb_risk_model")

# -----------------------------
# Save Model Locally
# -----------------------------
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"[INFO] Model saved at {MODEL_PATH}")
