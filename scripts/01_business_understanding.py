"""
CRISP-DM Phase 1: Business Understanding

Project:
PaySim Fraud Detection Using Feature Engineering
and Random Forest

Problem:
Financial transaction fraud is difficult to detect because
fraudulent transactions represent a very small proportion
of all transactions.

Project Objective:
Develop and evaluate a machine-learning approach for
identifying fraudulent PaySim transactions using Random
Forest classification and feature selection.

Research Questions:
1. Which transaction characteristics are the most important
   predictors of fraudulent transactions?

2. Does feature selection using Recursive Feature Elimination
   (RFE) maintain or improve Random Forest predictive
   performance?

Target Variable:
isFraud

Model:
Random Forest Classifier

Feature Selection:
Recursive Feature Elimination (RFE)

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Success Criterion:
Compare the baseline Random Forest with the RFE-based
Random Forest to determine whether a smaller feature set
can retain strong fraud-detection performance.
"""


def display_project_objectives():
    """Display the project's main analytical objective."""

    print("PaySim Fraud Detection Project")
    print("-" * 40)

    print(
        "Objective: Detect fraudulent financial transactions "
        "using Random Forest classification."
    )

    print(
        "Comparison: Baseline Random Forest vs. "
        "RFE Random Forest."
    )


if __name__ == "__main__":
    display_project_objectives()
