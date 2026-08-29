import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. Load data
df = pd.read_csv("Credit_Score_Classification_Updated.csv")

# 2. Drop identifiers / non-predictive columns
cols_to_drop = ["Customer Code"]
# Drop directly leaking/trivial features if needed
cols_to_drop.extend([col for col in ["Payment History", "Debt"] if col in df.columns])
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 3. Separate features and target
target_col = "Credit Score"
X = df.drop(columns=[target_col])
y = df[target_col]

# 4. Identify column types
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# 5. Preprocessing pipeline (One-Hot for nominals, Scaling for numerics)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

# 6. Define models wrapped in Pipelines to prevent data leakage
models = {
    "Logistic Regression": Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    ),
    "Decision Tree": Pipeline(
        [
            ("prep", preprocessor),
            ("clf", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100, max_depth=6, class_weight="balanced", random_state=42
                ),
            ),
        ]
    ),
}

# 7. Stratified Cross-Validation (Addresses small sample size)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]

print("=== Cross-Validation Results ===")
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    print(f"\nModel: {name}")
    print(f"  Accuracy : {scores['test_accuracy'].mean():.4f} (± {scores['test_accuracy'].std():.4f})")
    print(f"  F1-Score : {scores['test_f1_weighted'].mean():.4f} (± {scores['test_f1_weighted'].std():.4f})")

# 8. Train/Test evaluation for confusion matrix & report
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_model = models["Random Forest"]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\n=== Final Test Set Evaluation (Random Forest) ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
