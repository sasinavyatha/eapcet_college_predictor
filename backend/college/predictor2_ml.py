# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder

# # =========================
# # 1. LOAD DATA
# # =========================
# df = pd.read_csv("final_dataset.csv")

# # Clean text
# df['CATEGORY'] = df['CATEGORY'].str.strip().str.upper()
# df['GENDER'] = df['GENDER'].str.strip().str.upper()
# df['INST_NAME'] = df['INST_NAME'].str.strip()
# df['DISTRICT'] = df['DISTRICT'].str.strip().str.upper()
# df['BRANCH_CODE'] = df['BRANCH_CODE'].str.strip().str.upper()

# # =========================
# # 2. ENCODING
# # =========================
# le_category = LabelEncoder()
# le_gender = LabelEncoder()
# le_college = LabelEncoder()
# le_branch = LabelEncoder()

# df['CAT_ENC'] = le_category.fit_transform(df['CATEGORY'])
# df['GEN_ENC'] = le_gender.fit_transform(df['GENDER'])
# df['COL_ENC'] = le_college.fit_transform(df['INST_NAME'])
# df['BR_ENC'] = le_branch.fit_transform(df['BRANCH_CODE'])

# # =========================
# # 3. FEATURES
# # =========================
# X = df[['CAT_ENC', 'GEN_ENC', 'COL_ENC', 'BR_ENC', 'YEAR']]
# y = df['CUTOFF']

# # =========================
# # 4. MODEL
# # =========================
# model = RandomForestRegressor(n_estimators=50, random_state=42)
# model.fit(X, y)

# # =========================
# # 5. USER INPUT
# # =========================
# rank = int(input("Enter your rank: "))
# category = input("Enter category: ").strip().upper()
# gender = input("Enter gender: ").strip().upper()

# if gender == "MALE":
#     gender = "BOYS"
# elif gender == "FEMALE":
#     gender = "GIRLS"

# # Validate
# if category not in le_category.classes_:
#     print("❌ Invalid category:", list(le_category.classes_))
#     exit()

# if gender not in le_gender.classes_:
#     print("❌ Invalid gender:", list(le_gender.classes_))
#     exit()

# cat_enc = le_category.transform([category])[0]
# gen_enc = le_gender.transform([gender])[0]

# # =========================
# # 6. FILTER INPUTS (FIXED)
# # =========================
# print("\nAvailable Districts:", sorted(df['DISTRICT'].unique()))
# print("Available Branches:", sorted(df['BRANCH_CODE'].unique()))

# district_input = input("Enter districts (comma-separated or ALL): ").strip().upper()
# branch_input = input("Enter branches (comma-separated or ALL): ").strip().upper()

# if district_input == "" or district_input == "ALL":
#     selected_districts = None
# else:
#     selected_districts = [d.strip() for d in district_input.split(",")]

# if branch_input == "" or branch_input == "ALL":
#     selected_branches = None
# else:
#     selected_branches = [b.strip() for b in branch_input.split(",")]

# # =========================
# # 7. PREDICTION
# # =========================
# results = []
# latest_year = df['YEAR'].max()

# unique_college_branch = df[['INST_NAME', 'BRANCH_CODE']].drop_duplicates()

# for _, row in unique_college_branch.iterrows():
#     col = row['INST_NAME']
#     br = row['BRANCH_CODE']

#     # Get district
#     district = df[df['INST_NAME'] == col]['DISTRICT'].iloc[0]

#     # 🔴 FILTER 1: District
#     if selected_districts and district not in selected_districts:
#         continue

#     # 🔴 FILTER 2: Branch
#     if selected_branches and br not in selected_branches:
#         continue

#     # 🔴 FILTER 3: Women colleges restriction
#     if gender == "BOYS" and "WOMEN" in col.upper():
#         continue

#     col_enc = le_college.transform([col])[0]
#     br_enc = le_branch.transform([br])[0]

#     input_df = pd.DataFrame([{
#         'CAT_ENC': cat_enc,
#         'GEN_ENC': gen_enc,
#         'COL_ENC': col_enc,
#         'BR_ENC': br_enc,
#         'YEAR': latest_year
#     }])

#     # =========================
#     # HYBRID PREDICTION
#     # =========================
#     ml_cutoff = model.predict(input_df)[0]

#     last_year_data = df[
#         (df['INST_NAME'] == col) &
#         (df['BRANCH_CODE'] == br) &
#         (df['CATEGORY'] == category) &
#         (df['GENDER'] == gender)
#     ]

#     last_year_cutoff = last_year_data['CUTOFF'].max()

#     if pd.notna(last_year_cutoff):
#         pred_cutoff = (0.85 * last_year_cutoff) + (0.15 * ml_cutoff)
#     else:
#         pred_cutoff = ml_cutoff

#     # =========================
#     # REALISTIC FILTERING
#     # =========================
#     if rank > pred_cutoff + 1500:
#         continue

#     # =========================
#     # SAFE / TARGET / DREAM
#     # =========================
#     if rank <= pred_cutoff * 0.9:
#         chance = "SAFE"
#     elif rank <= pred_cutoff:
#         chance = "TARGET"
#     else:
#         chance = "DREAM"

#     # =========================
#     # CONFIDENCE SCORE
#     # =========================
#     confidence = round(100 - abs(rank - pred_cutoff) / pred_cutoff * 100, 2)

#     results.append([col, br, district, int(pred_cutoff), chance, confidence])

# # =========================
# # 8. OUTPUT
# # =========================
# if len(results) == 0:
#     print("\n⚠️ No colleges found. Try different filters.")
#     exit()

# result_df = pd.DataFrame(results, columns=[
#     "College", "Branch", "District", "Predicted_Cutoff", "Chance", "Confidence"
# ])

# # Remove duplicates
# result_df = result_df.sort_values(by='Predicted_Cutoff')
# result_df = result_df.drop_duplicates(subset=['College', 'Branch'], keep='first')

# # Sorting
# sort_choice = input("Sort by (1: Chance, 2: Cutoff): ").strip()

# chance_order = {"SAFE": 1, "TARGET": 2, "DREAM": 3}

# if sort_choice == "1":
#     result_df['sort'] = result_df['Chance'].map(chance_order)
#     result_df = result_df.sort_values(by=['sort', 'Predicted_Cutoff'])
# else:
#     result_df = result_df.sort_values(by=['Predicted_Cutoff'])

# # Optional post-filter
# post_filter = input("Show only SAFE/TARGET? (yes/no): ").strip().lower()

# if post_filter == "yes":
#     result_df = result_df[result_df['Chance'] != "DREAM"]

# print("\n🎯 FINAL COLLEGE PREDICTIONS:\n")
# print(result_df.head(20))

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("final_dataset.csv")

# Clean text
df['CATEGORY'] = df['CATEGORY'].str.strip().str.upper()
df['GENDER'] = df['GENDER'].str.strip().str.upper()
df['INST_NAME'] = df['INST_NAME'].str.strip()
df['DISTRICT'] = df['DISTRICT'].str.strip().str.upper()
df['BRANCH_CODE'] = df['BRANCH_CODE'].str.strip().str.upper()

# =========================
# 2. ENCODING
# =========================
le_category = LabelEncoder()
le_gender = LabelEncoder()
le_college = LabelEncoder()
le_branch = LabelEncoder()

df['CAT_ENC'] = le_category.fit_transform(df['CATEGORY'])
df['GEN_ENC'] = le_gender.fit_transform(df['GENDER'])
df['COL_ENC'] = le_college.fit_transform(df['INST_NAME'])
df['BR_ENC'] = le_branch.fit_transform(df['BRANCH_CODE'])

# =========================
# 3. FEATURES
# =========================
X = df[['CAT_ENC', 'GEN_ENC', 'COL_ENC', 'BR_ENC', 'YEAR']]
y = df['CUTOFF']

# =========================
# 4. MODEL
# =========================
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

# =========================
# 5. USER INPUT
# =========================
rank = int(input("Enter your rank: "))
category = input("Enter category: ").strip().upper()
gender = input("Enter gender: ").strip().upper()

if gender == "MALE":
    gender = "BOYS"
elif gender == "FEMALE":
    gender = "GIRLS"

# Validate
if category not in le_category.classes_:
    print("❌ Invalid category:", list(le_category.classes_))
    exit()

if gender not in le_gender.classes_:
    print("❌ Invalid gender:", list(le_gender.classes_))
    exit()

cat_enc = le_category.transform([category])[0]
gen_enc = le_gender.transform([gender])[0]

# =========================
# 6. FILTER INPUTS
# =========================
print("\nAvailable Districts:", sorted(df['DISTRICT'].unique()))
print("Available Branches:", sorted(df['BRANCH_CODE'].unique()))

district_input = input("Enter districts (comma-separated or ALL): ").strip().upper()
branch_input = input("Enter branches (comma-separated or ALL): ").strip().upper()

if district_input == "" or district_input == "ALL":
    selected_districts = None
else:
    selected_districts = [d.strip() for d in district_input.split(",")]

if branch_input == "" or branch_input == "ALL":
    selected_branches = None
else:
    selected_branches = [b.strip() for b in branch_input.split(",")]

# =========================
# 7. PREDICTION
# =========================
results = []
latest_year = df['YEAR'].max()

unique_college_branch = df[['INST_NAME', 'BRANCH_CODE']].drop_duplicates()

for _, row in unique_college_branch.iterrows():
    col = row['INST_NAME']
    br = row['BRANCH_CODE']

    # Get district
    district = df[df['INST_NAME'] == col]['DISTRICT'].iloc[0]

    # 🔴 FILTER 1: District
    if selected_districts and district not in selected_districts:
        continue

    # 🔴 FILTER 2: Branch
    if selected_branches and br not in selected_branches:
        continue

    # 🔴 FILTER 3: Women colleges restriction
    if gender == "BOYS" and "WOMEN" in col.upper():
        continue

    col_enc = le_college.transform([col])[0]
    br_enc = le_branch.transform([br])[0]

    input_df = pd.DataFrame([{
        'CAT_ENC': cat_enc,
        'GEN_ENC': gen_enc,
        'COL_ENC': col_enc,
        'BR_ENC': br_enc,
        'YEAR': latest_year
    }])

    # =========================
    # HYBRID PREDICTION (IMPROVED)
    # =========================
    ml_cutoff = model.predict(input_df)[0]

    # Last 3 years data
    last_3_years = df[
        (df['INST_NAME'] == col) &
        (df['BRANCH_CODE'] == br) &
        (df['CATEGORY'] == category) &
        (df['GENDER'] == gender)
    ].sort_values(by='YEAR', ascending=False).head(3)

    avg_cutoff = last_3_years['CUTOFF'].mean()

    if pd.notna(avg_cutoff):
        pred_cutoff = (0.8 * avg_cutoff) + (0.2 * ml_cutoff)
    else:
        pred_cutoff = ml_cutoff

    # 🔥 Calibration (fix overestimation)
    pred_cutoff = pred_cutoff * 0.95

    # =========================
    # REALISTIC FILTERING
    # =========================
    if rank > pred_cutoff + 1500:
        continue

    # =========================
    # SAFE / TARGET / DREAM
    # =========================
    if rank <= pred_cutoff - 1500:
        chance = "SAFE"
    elif rank <= pred_cutoff:
        chance = "TARGET"
    else:
        chance = "DREAM"

    # =========================
    # CONFIDENCE SCORE
    # =========================
    confidence = round(100 - abs(rank - pred_cutoff) / pred_cutoff * 100, 2)

    results.append([col, br, district, int(pred_cutoff), chance, confidence])

# =========================
# 8. OUTPUT
# =========================
if len(results) == 0:
    print("\n⚠️ No colleges found. Try different filters.")
    exit()

result_df = pd.DataFrame(results, columns=[
    "College", "Branch", "District", "Predicted_Cutoff", "Chance", "Confidence"
])

# Remove duplicates
result_df = result_df.sort_values(by='Predicted_Cutoff')
result_df = result_df.drop_duplicates(subset=['College', 'Branch'], keep='first')

# Sorting
sort_choice = input("Sort by (1: Chance, 2: Cutoff): ").strip()

chance_order = {"SAFE": 1, "TARGET": 2, "DREAM": 3}

if sort_choice == "1":
    result_df['sort'] = result_df['Chance'].map(chance_order)
    result_df = result_df.sort_values(by=['sort', 'Predicted_Cutoff'])
else:
    result_df = result_df.sort_values(by=['Predicted_Cutoff'])

# Optional post-filter
post_filter = input("Show only SAFE/TARGET? (yes/no): ").strip().lower()

if post_filter == "yes":
    result_df = result_df[result_df['Chance'] != "DREAM"]

print("\n🎯 FINAL COLLEGE PREDICTIONS:\n")
print(result_df.head(20))