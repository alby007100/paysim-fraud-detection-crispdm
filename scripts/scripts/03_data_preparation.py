import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Fraud_Detection_Dataset.csv"

# Load the dataset
data = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

df_fe = data.copy()

# Balance-change features
df_fe["balanceChangeOrig"] = (
    df_fe["oldbalanceOrg"] - df_fe["newbalanceOrig"]
)

df_fe["balanceChangeDest"] = (
    df_fe["newbalanceDest"] - df_fe["oldbalanceDest"]
)

# Transaction amount relative to origin balance
df_fe["transactionRatio"] = (
    df_fe["amount"] / (df_fe["oldbalanceOrg"] + 1)
)

# Temporal features
df_fe["hourOfDay"] = df_fe["step"] % 24
df_fe["day"] = df_fe["step"] // 24

# Large-transaction indicator
df_fe["isLargeTransaction"] = (
    df_fe["amount"] > df_fe["amount"].quantile(0.95)
).astype(int)

# --------------------------------------------------
# Prepare Modeling Dataset
# --------------------------------------------------

# Remove account identifiers
df_model = df_fe.drop(
    columns=["nameOrig", "nameDest"]
)

# Convert transaction type into dummy variables
df_model = pd.get_dummies(
    df_model,
    columns=["type"],
    drop_first=True
)

# Separate predictors and target
X = df_model.drop(columns=["isFraud"])
y = df_model["isFraud"]

print("Number of predictors:", X.shape[1])
print("\nTarget distribution:")
print(y.value_counts())

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining observations:", X_train.shape[0])
print("Testing observations:", X_test.shape[0])
print("Number of features:", X_train.shape[1])
