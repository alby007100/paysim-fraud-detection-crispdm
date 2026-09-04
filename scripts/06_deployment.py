import joblib
from pathlib import Path


# --------------------------------------------------
# Define Output Directory
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Save Final RFE Random Forest Model
# --------------------------------------------------

MODEL_PATH = OUTPUT_DIR / "rfe_random_forest_model.pkl"

joblib.dump(
    rfe_rf,
    MODEL_PATH
)

print(f"Model saved to: {MODEL_PATH}")


# --------------------------------------------------
# Save Selected Features
# --------------------------------------------------

FEATURE_PATH = OUTPUT_DIR / "selected_features.txt"

with open(FEATURE_PATH, "w") as file:
    for feature in selected_features_rfe:
        file.write(feature + "\n")

print(f"Selected features saved to: {FEATURE_PATH}")


# --------------------------------------------------
# Deployment Information
# --------------------------------------------------

print("\nDeployment completed successfully.")
print("The trained Random Forest model can now be")
print("loaded and used to classify new transactions.")
