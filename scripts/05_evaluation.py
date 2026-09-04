import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"


# --------------------------------------------------
# Load Models and Test Data
# --------------------------------------------------

baseline_rf = joblib.load(
    MODEL_DIR / "baseline_random_forest.joblib"
)

rfe_rf = joblib.load(
    MODEL_DIR / "rfe_random_forest.joblib"
)

X_test = joblib.load(
    OUTPUT_DIR / "X_test.joblib"
)

X_test_rfe = joblib.load(
    OUTPUT_DIR / "X_test_rfe.joblib"
)

y_test = joblib.load(
    OUTPUT_DIR / "y_test.joblib"
)


# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------

def evaluate_model(model, X, y, model_name):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions),
        "Recall": recall_score(y, predictions),
        "F1-score": f1_score(y, predictions),
        "ROC-AUC": roc_auc_score(y, probabilities),
    }

    print(f"\n{model_name}")
    print("-" * 50)

    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))

    return metrics


# --------------------------------------------------
# Evaluate Models
# --------------------------------------------------

baseline_metrics = evaluate_model(
    baseline_rf,
    X_test,
    y_test,
    "Baseline Random Forest"
)

rfe_metrics = evaluate_model(
    rfe_rf,
    X_test_rfe,
    y_test,
    "RFE Random Forest"
)


# --------------------------------------------------
# Save Evaluation Results
# --------------------------------------------------

results = pd.DataFrame([
    baseline_metrics,
    rfe_metrics
])

RESULTS_PATH = OUTPUT_DIR / "evaluation_metrics.csv"

results.to_csv(
    RESULTS_PATH,
    index=False
)

print("\nEvaluation Summary")
print(results)

print(
    f"\nEvaluation metrics saved to: {RESULTS_PATH}"
)
