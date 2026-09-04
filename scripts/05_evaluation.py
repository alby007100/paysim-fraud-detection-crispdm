import joblib
from pathlib import Path


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"


# --------------------------------------------------
# Load Final Model
# --------------------------------------------------

MODEL_PATH = MODEL_DIR / "rfe_random_forest.joblib"

final_model = joblib.load(MODEL_PATH)

print(f"Final model loaded from: {MODEL_PATH}")


# --------------------------------------------------
# Load Selected Features
# --------------------------------------------------

FEATURES_PATH = OUTPUT_DIR / "selected_features_rfe.joblib"

selected_features = joblib.load(FEATURES_PATH)

print("\nSelected features used by the final model:")

for feature in selected_features:
    print("-", feature)


# --------------------------------------------------
# Deployment Summary
# --------------------------------------------------

print("\nDeployment stage completed successfully.")

print(
    "The trained RFE Random Forest model is ready "
    "to be used for scoring new transactions."
)
