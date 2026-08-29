import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

df = pd.read_csv("Credit_Score_Classification_Updated.csv")

# Clean & separate target
cols_to_drop = ["Customer Code"]
cols_to_drop.extend([col for col in ["Payment History", "Debt"] if col in df.columns])
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

target_col = "Credit Score"
df = df.dropna(subset=[target_col])
X = df.drop(columns=[target_col])
y = df[target_col]

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

for c in num_cols:
    X[c] = X[c].fillna(X[c].median())
for c in cat_cols:
    X[c] = X[c].fillna("Missing")

# Full pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42))
])

pipeline.fit(X, y)

# Save the full model artifact with metadata
model_data = {
    "pipeline": pipeline,
    "feature_cols": list(X.columns),
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "feature_summary": {
        col: {"min": float(X[col].min()), "max": float(X[col].max()), "mean": float(X[col].mean())}
        if col in num_cols else sorted(list(set(X[col].astype(str))))
        for col in X.columns
    }
}

joblib.dump(model_data, "credit_model.pkl")
print("Saved model and metadata to credit_model.pkl")
