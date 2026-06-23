# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder

# # =========================
# # 1. LOAD DATA
# # =========================
# df = pd.read_csv("final_dataset.csv")

# # =========================
# # 2. PREPROCESSING
# # =========================

# # Clean text
# df['CATEGORY'] = df['CATEGORY'].str.strip().str.upper()
# df['GENDER'] = df['GENDER'].str.strip().str.upper()

# # Encode
# le_category = LabelEncoder()
# le_gender = LabelEncoder()

# df['CATEGORY_ENC'] = le_category.fit_transform(df['CATEGORY'])
# df['GENDER_ENC'] = le_gender.fit_transform(df['GENDER'])

# # =========================
# # 3. TRAIN MODEL (optional but kept for interview)
# # =========================
# X = df[['CATEGORY_ENC', 'GENDER_ENC']]
# y = df['CUTOFF']

# model = RandomForestRegressor(n_estimators=50, random_state=42)
# model.fit(X, y)

# # =========================
# # 4. USER INPUT
# # =========================
# rank = int(input("Enter your rank: "))

# category = input("Enter category: ").strip().upper()
# gender = input("Enter gender: ").strip().upper()

# # Fix gender mapping
# if gender == "MALE":
#     gender = "BOYS"
# elif gender == "FEMALE":
#     gender = "GIRLS"

# # =========================
# # 5. VALIDATE INPUT
# # =========================
# if category not in le_category.classes_:
#     print("❌ Invalid category. Allowed:", list(le_category.classes_))
#     exit()

# if gender not in le_gender.classes_:
#     print("❌ Invalid gender. Allowed:", list(le_gender.classes_))
#     exit()

# # Encode input
# cat_enc = le_category.transform([category])[0]
# gen_enc = le_gender.transform([gender])[0]

# # =========================
# # 6. FAST FILTERING (NO LOOP)
# # =========================

# # Filter near range
# filtered_df = df[df['CUTOFF'] >= rank - 5000].copy()

# # =========================
# # 7. CHANCE LOGIC
# # =========================
# def get_chance(cutoff):
#     if rank <= cutoff:
#         return "High"
#     elif rank <= cutoff + 2000:
#         return "Medium"
#     else:
#         return "Low"

# filtered_df['Chance'] = filtered_df['CUTOFF'].apply(get_chance)

# # =========================
# # 8. FINAL RESULT
# # =========================
# result_df = filtered_df[['INST_NAME', 'BRANCH_CODE', 'CUTOFF', 'Chance']].copy()
# result_df.columns = ["College", "Branch", "Predicted_Cutoff", "Chance"]

# # Remove duplicates
# result_df = result_df.drop_duplicates()

# # Sort results
# chance_order = {"High": 1, "Medium": 2, "Low": 3}
# result_df['sort'] = result_df['Chance'].map(chance_order)

# result_df = result_df.sort_values(by=['sort', 'Predicted_Cutoff'])

# # =========================
# # 9. OUTPUT
# # =========================
# if result_df.empty:
#     print("\n⚠️ No colleges found. Try different rank range.")
# else:
#     print("\n🎯 Top Results:\n")
#     print(result_df[['College', 'Branch', 'Predicted_Cutoff', 'Chance']].head(100))


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
# # 5. INPUT
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
#     print("Invalid category")
#     exit()

# if gender not in le_gender.classes_:
#     print("Invalid gender")
#     exit()

# cat_enc = le_category.transform([category])[0]
# gen_enc = le_gender.transform([gender])[0]

# # =========================
# # 6. PREDICT FOR ALL COLLEGES
# # =========================
# results = []

# latest_year = df['YEAR'].max()

# unique_college_branch = df[['INST_NAME', 'BRANCH_CODE']].drop_duplicates()

# for _, row in unique_college_branch.iterrows():
#     col = row['INST_NAME']
#     br = row['BRANCH_CODE']

#     col_enc = le_college.transform([col])[0]
#     br_enc = le_branch.transform([br])[0]

#     input_df = pd.DataFrame([{
#         'CAT_ENC': cat_enc,
#         'GEN_ENC': gen_enc,
#         'COL_ENC': col_enc,
#         'BR_ENC': br_enc,
#         'YEAR': latest_year
#     }])

#     pred_cutoff = model.predict(input_df)[0]

#     if rank <= pred_cutoff + 3000:
#         if rank <= pred_cutoff:
#             chance = "High"
#         elif rank <= pred_cutoff + 1500:
#             chance = "Medium"
#         else:
#             chance = "Low"

#         results.append([col, br, int(pred_cutoff), chance])

# # =========================
# # 7. OUTPUT
# # =========================
# result_df = pd.DataFrame(results, columns=[
#     "College", "Branch", "Predicted_Cutoff", "Chance"
# ])

# result_df = result_df.drop_duplicates()

# chance_order = {"High": 1, "Medium": 2, "Low": 3}
# result_df['sort'] = result_df['Chance'].map(chance_order)

# result_df = result_df.sort_values(by=['sort', 'Predicted_Cutoff'])

# print("\n🎯 Top Results:\n")
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
# 5. INPUT
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
# FILTER INPUTS
# =========================

# =========================
# FILTER INPUTS (FIXED)
# =========================

district_input = input("Enter districts (comma-separated or ALL): ").strip().upper()
branch_input = input("Enter branches (comma-separated or ALL): ").strip().upper()

# ✅ Handle district properly
if district_input == "" or district_input == "ALL":
    selected_districts = None
else:
    selected_districts = [d.strip() for d in district_input.split(",")]

# ✅ Handle branch properly
if branch_input == "" or branch_input == "ALL":
    selected_branches = None
else:
    selected_branches = [b.strip() for b in branch_input.split(",")]
# =========================
# 6. PREDICT FOR ALL COLLEGES
# =========================
results = []
latest_year = df['YEAR'].max()

unique_college_branch = df[['INST_NAME', 'BRANCH_CODE']].drop_duplicates()

for _, row in unique_college_branch.iterrows():
    col = row['INST_NAME']
    br = row['BRANCH_CODE']

    # Get district
    district = df[df['INST_NAME'] == col]['DISTRICT'].iloc[0].upper()

    # 🔴 FILTER 1: District filter
    if selected_districts and district not in selected_districts:
        continue

    # 🔴 FILTER 2: Branch filter
    if selected_branches and br not in selected_branches:
        continue

    # 🔴 FILTER 3: Women college restriction
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

    pred_cutoff = model.predict(input_df)[0]

    if rank <= pred_cutoff + 2000:

        if rank <= pred_cutoff:
            chance = "High"
        elif rank <= pred_cutoff + 1000:
            chance = "Medium"
        else:
            chance = "Low"

        results.append([col, br, district, int(pred_cutoff), chance])
# =========================
# 7. OUTPUT
# =========================
if len(results) == 0:
    print("\n⚠️ No colleges found. Try different rank/category.")
    exit()

result_df = pd.DataFrame(results, columns=[
    "College", "Branch", "District", "Predicted_Cutoff", "Chance"
])

# Remove duplicates
result_df = result_df.drop_duplicates()

# Sorting
sort_choice = input("Sort by (1: Chance, 2: Cutoff): ").strip()

chance_order = {"High": 1, "Medium": 2, "Low": 3}

if sort_choice == "1":
    result_df['sort'] = result_df['Chance'].map(chance_order)
    result_df = result_df.sort_values(by=['sort', 'Predicted_Cutoff'])
else:
    result_df = result_df.sort_values(by=['Predicted_Cutoff'])

print("\n🎯 Top Results:\n")
print(result_df.head(100))
filter_after = input("Show only High/Medium chances? (yes/no): ").strip().lower()

if filter_after == "yes":
    result_df = result_df[result_df['Chance'] != "Low"]