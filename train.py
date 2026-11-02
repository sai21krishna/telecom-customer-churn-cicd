
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from app.preprocessing import preprocess_data


# Load dataset
df = pd.read_excel("data/customer_churn.xlsx")

print("Original dataset shape:", df.shape)

# Separate features and target
X = preprocess_data(df)

y = df["Churn"].map({
    "Not Churn": 0,
    "Churn": 1
})

if y.isna().any():
    raise ValueError("Unexpected or missing target values")

print("Preprocessed features:", X.shape)
print("\nTarget distribution:")
print(y.value_counts())


# Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


# Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# Define two models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )
}


results = {}
trained_models = {}

# Train and evaluate
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results[name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }

    trained_models[name] = pipeline

    print("\nMODEL:", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Not Churn", "Churn"],
        zero_division=0
    ))

# Select best model using F1-score
best_model_name = max(
    results,
    key=lambda name: results[name]["f1_score"]
)

best_model = trained_models[best_model_name]

# Save model and metrics
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/churn_model.pkl")

with open("models/metrics.json", "w") as file:
    json.dump({
        "best_model": best_model_name,
        "results": results
    }, file, indent=4)

print("\n================================")
print("BEST MODEL:", best_model_name)
print("MODEL SAVED SUCCESSFULLY")
print("================================")

print(json.dumps(results, indent=4))
