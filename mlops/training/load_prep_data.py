import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
RAW_DATA_DIR = Path("data/raw/cicids2017")
PROCESSED_DATA_PATH = Path("data/processed/cicids2017_v1.csv")

print("[INFO] Looking for CSV files...")

csv_files = list(RAW_DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found in data/raw/cicids2017")

print(f"[INFO] Found {len(csv_files)} files")

df_list = []

for file in csv_files:
    print(f"[INFO] Loading: {file.name}")

    try:
        temp_df = pd.read_csv(
            file,
            low_memory=False,
            encoding="latin1"
        )
    except Exception as e:
        print(f"[WARNING] Skipping {file.name} due to error: {e}")
        continue

    temp_df.columns = temp_df.columns.str.strip()

    df_list.append(temp_df)

# -----------------------------
# Merge All Files
# -----------------------------
print("[INFO] Concatenating files...")
df = pd.concat(df_list, ignore_index=True)

print("[INFO] Combined shape:", df.shape)

# -----------------------------
# Label Handling
# -----------------------------
if "Label" not in df.columns:
    raise ValueError("Label column not found. Check column names.")

df["target"] = df["Label"].apply(lambda x: 0 if x.strip() == "BENIGN" else 1)

# -----------------------------
# Drop Identifier Columns
# -----------------------------
DROP_COLS = [
    "Label",
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp"
]

df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors="ignore")

# -----------------------------
# Handle Bad Values
# -----------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# -----------------------------
# Final Check
# -----------------------------
print("[INFO] Final shape:", df.shape)
print("[INFO] Target distribution:")
print(df["target"].value_counts(normalize=True))

# -----------------------------
# Save Processed File
# -----------------------------
PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(PROCESSED_DATA_PATH, index=False)

print(f"[INFO] Saved to {PROCESSED_DATA_PATH}")
