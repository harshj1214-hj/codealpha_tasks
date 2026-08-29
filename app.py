import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Credit Score Classifier", page_icon="💳", layout="centered")

CSV_FILENAME = "Credit_Score_Classification_Updated.csv"

@st.cache_resource
def train_model():
    if not os.path.exists(CSV_FILENAME):
        return None, None, None, None, f"Dataset file '{CSV_FILENAME}' not found in root directory."

    df = pd.read_csv(CSV_FILENAME)
    
    # Drop known identifiers and trivial columns
    cols_to_drop = ["Customer Code"]
    cols_to_drop.extend([col for col in ["Payment History", "Debt"] if col in df.columns])
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    
    target_col = "Credit Score"
    if target_col not in df.columns:
        return None, None, None, None, f"Target column '{target_col}' not found in CSV."

    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Categorize column types safely
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fill missing values for pipeline robustness
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].fillna("Missing")

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
    return pipeline, X, cat_cols, num_cols, None

pipeline, X_raw, cat_cols, num_cols, error_msg = train_model()

if error_msg:
    st.error(f"Error loading model: {error_msg}")
    st.stop()

st.title("💳 Credit Score Prediction App")
st.markdown("Enter applicant details to predict credit score category.")

inputs = {}
st.subheader("Applicant Details")
col1, col2 = st.columns(2)

for i, col in enumerate(X_raw.columns):
    target_col_container = col1 if i % 2 == 0 else col2
    with target_col_container:
        if col in num_cols:
            min_v = float(X_raw[col].min())
            max_v = float(X_raw[col].max())
            mean_v = float(X_raw[col].mean())
            
            # Prevent min == max crash
            if min_v >= max_v:
                max_v = min_v + 1.0
            
            inputs[col] = st.number_input(f"{col}", min_value=min_v, max_value=max_v, value=mean_v)
        else:
            unique_opts = sorted(list(set(X_raw[col].dropna().astype(str).tolist())))
            inputs[col] = st.selectbox(f"{col}", options=unique_opts)

if st.button("Predict Credit Score", type="primary"):
    input_df = pd.DataFrame([inputs])
    prediction = pipeline.predict(input_df)[0]
    st.success(f"**Predicted Credit Score:** {prediction}")
