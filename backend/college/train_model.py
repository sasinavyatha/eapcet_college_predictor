import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("final_dataset.csv")

# Encode categorical columns
le_inst = LabelEncoder()
le_branch = LabelEncoder()
le_cat = LabelEncoder()
le_gender = LabelEncoder()

df["INST_CODE"] = le_inst.fit_transform(df["INST_CODE"])
df["BRANCH_CODE"] = le_branch.fit_transform(df["BRANCH_CODE"])
df["CATEGORY"] = le_cat.fit_transform(df["CATEGORY"])
df["GENDER"] = le_gender.fit_transform(df["GENDER"])

# Features & target
X = df[["INST_CODE", "BRANCH_CODE", "CATEGORY", "GENDER", "YEAR"]]
y = df["CUTOFF"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model + encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump((le_inst, le_branch, le_cat, le_gender), open("encoders.pkl", "wb"))

print("Model trained and saved ✅")