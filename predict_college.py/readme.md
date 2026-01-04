# ==========================================
# TOP 100 COLLEGES NEAREST TO USER RANK
# (WITH PROBABILITY DISPLAY)
# ==========================================

import pandas as pd
import numpy as np
import joblib

# ==========================================
# LOAD MODEL & ENCODERS
# ==========================================
model = joblib.load("college_xgb_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# ==========================================
# LOAD DATASET
# ==========================================
df = pd.read_csv("final_eapcet_dataset.csv")
df.columns = df.columns.str.strip().str.replace(" ", "", regex=False)

# Clean college names
df["inst_name"] = (
    df["inst_name"]
    .str.replace("\n", " ", regex=False)
    .str.replace("  ", " ")
    .str.strip()
)

# ==========================================
# USER INPUT
# ==========================================
user_rank = int(input("Enter your rank: "))
branch = input("Enter branch code (e.g. AGR, CSE): ")
category = input("Enter category (OC/SC/ST/BCA/BCB): ")
gender = input("Enter gender (BOYS/GIRLS): ")
year = int(input("Enter admission year (e.g. 2025): "))

RISK_TOLERANCE = 5000
PROB_SCALE = 15000   # controls probability decay

# ==========================================
# HELPER: DISTANCE → PROBABILITY
# ==========================================
def distance_to_probability(distance):
    prob = 100 * np.exp(-abs(distance) / PROB_SCALE)
    return round(prob, 2)

# ==========================================
# PREDICT FOR ALL COLLEGES
# ==========================================
rows = []

for _, row in df.iterrows():
    input_row = {
        "inst_code": row["inst_code"],
        "type": row["type"],
        "DIST": row["DIST"],
        "PLACE": row["PLACE"],
        "COED": row["COED"],
        "branch_code": branch,
        "Year": year,
        "gender": gender,
        "category": category
    }

    df_input = pd.DataFrame([input_row])

    # Encode categorical columns
    for col, enc in encoders.items():
        if col in df_input.columns:
            if df_input[col].iloc[0] in enc.classes_:
                df_input[col] = enc.transform(df_input[col])
            else:
                df_input[col] = enc.transform([enc.classes_[0]])

    # EXACT training feature order
    X_input = df_input[
        [
            "inst_code",
            "type",
            "DIST",
            "PLACE",
            "COED",
            "branch_code",
            "Year",
            "gender",
            "category"
        ]
    ]

    pred_log = model.predict(X_input)[0]
    pred_cutoff = int(np.expm1(pred_log))

    rows.append({
        "College": row["inst_name"],
        "District": row["DIST"],
        "Predicted_Cutoff": pred_cutoff
    })

# ==========================================
# AGGREGATE UNIQUE COLLEGES
# ==========================================
result_df = pd.DataFrame(rows)

result_df = (
    result_df
    .groupby("College", as_index=False)
    .agg({
        "District": "first",
        "Predicted_Cutoff": "max"
    })
)

# ==========================================
# DISTANCE-BASED RANKING (CORE LOGIC)
# ==========================================
result_df["Distance"] = result_df["Predicted_Cutoff"] - user_rank

possible = result_df[result_df["Distance"] >= 0]

if len(possible) < 100:
    risky = result_df[
        (result_df["Distance"] < 0) &
        (result_df["Distance"] >= -RISK_TOLERANCE)
    ]
    possible = pd.concat([possible, risky])

possible["Abs_Distance"] = possible["Distance"].abs()
possible = possible.sort_values("Abs_Distance")

top100 = possible.head(100)

# ==========================================
# ADD PROBABILITY (DISPLAY ONLY)
# ==========================================
top100["Probability (%)"] = top100["Distance"].apply(distance_to_probability)

# ==========================================
# DISPLAY RESULT
# ==========================================
print("\n===== TOP 100 COLLEGES NEAREST TO YOUR RANK =====\n")
print(
    top100[
        ["College", "District", "Predicted_Cutoff", "Probability (%)"]
    ].to_string(index=False)
)
