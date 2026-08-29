import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Credit Score Classifier", page_icon="💳", layout="centered")

@st.cache_resource
def train_model():
    df = pd.read_csv("Credit_Score_Classification_Updated.csv")
    
    cols_to_drop = ["Customer Code"]
    cols_to_drop.extend([col for col in ["Payment History", "Debt"] if col in df.columns])
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    target_col = "Credit Score"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
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
    return pipeline, X, cat_cols, num_cols

pipeline, X_raw, cat_cols, num_cols = train_model()

st.title("💳 Credit Score Prediction App")
st.markdown("Enter applicant details to predict credit score category.")

inputs = {}

st.subheader("Applicant Details")
col1, col2 = st.columns(2)

# Generate dynamic UI based on dataset features
for i, col in enumerate(X_raw.columns):
    target_col_container = col1 if i % 2 == 0 else col2
    with target_col_container:
        if col in num_cols:
            min_v = float(X_raw[col].min())
            max_v = float(X_raw[col].max())
            mean_v = float(X_raw[col].mean())
            inputs[col] = st.number_input(f"{col}", min_value=min_v, max_value=max_v, value=mean_v)
        else:
            unique_opts = sorted(X_raw[col].dropna().unique().tolist())
            inputs[col] = st.selectbox(f"{col}", options=unique_opts)

if st.button("Predict Credit Score", type="primary"):
    input_df = pd.DataFrame([inputs])
    prediction = pipeline.predict(input_df)[0]
    
    st.success(f"**Predicted Credit Score:** {prediction}")
