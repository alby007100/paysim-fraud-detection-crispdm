import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a classification model using the main
    performance metrics used in the assignment.
    """

    # Generate predictions
    y_pred = model.predict(X_test)

    # Generate fraud probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate performance metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    print(f"\n{model_name}")
    print("-" * 50)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


# --------------------------------------------------
# Evaluate Baseline Random Forest
# --------------------------------------------------

baseline_metrics = evaluate_model(
    baseline_rf,
    X_test,
    y_test,
    "Baseline Random Forest"
)


# --------------------------------------------------
# Evaluate RFE Random Forest
# --------------------------------------------------

rfe_metrics = evaluate_model(
    rfe_rf,
    X_test_rfe,
    y_test,
    "RFE Random Forest"
)


# --------------------------------------------------
# Compare Model Performance
# --------------------------------------------------

comparison = pd.DataFrame({
    "Baseline Random Forest": baseline_metrics,
    "RFE Random Forest": rfe_metrics
})

print("\nModel Performance Comparison")
print("=" * 50)

print(comparison)
