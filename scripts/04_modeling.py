import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Fraud_Detection_Dataset.csv"


# --------------------------------------------------
# Load Data
# --------------------------------------------------

data = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

df_fe = data.copy()

df_fe["balanceChangeOrig"] = (
    df_fe["oldbalanceOrg"] - df_fe["newbalanceOrig"]
)

df_fe["balanceChangeDest"] = (
    df_fe["newbalanceDest"] - df_fe["oldbalanceDest"]
)

df_fe["transactionRatio"] = (
    df_fe["amount"] / (df_fe["oldbalanceOrg"] + 1)
)

df_fe["hourOfDay"] = df_fe["step"] % 24
df_fe["day"] = df_fe["step"] // 24

df_fe["isLargeTransaction"] = (
    df_fe["amount"] > df_fe["amount"].quantile(0.95)
).astype(int)


# --------------------------------------------------
# Prepare Modeling Data
# --------------------------------------------------

df_model = df_fe.drop(
    columns=["nameOrig", "nameDest"]
)

df_model = pd.get_dummies(
    df_model,
    columns=["type"],
    drop_first=True
)

X = df_model.drop(columns=["isFraud"])
y = df_model["isFraud"]


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

print("Training observations:", X_train.shape[0])
print("Testing observations:", X_test.shape[0])
print("Number of baseline features:", X_train.shape[1])


# --------------------------------------------------
# Baseline Random Forest
# --------------------------------------------------

baseline_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

baseline_rf.fit(
    X_train,
    y_train
)

print("\nBaseline Random Forest trained successfully.")


# --------------------------------------------------
# Recursive Feature Elimination Sample
# --------------------------------------------------

# Keep all fraud observations from the training set
fraud_idx = y_train[y_train == 1].index

# Randomly sample 200,000 non-fraud observations
nonfraud_idx = y_train[y_train == 0].sample(
    n=200000,
    random_state=42
).index

# Combine fraud and non-fraud observations
rfe_idx = fraud_idx.union(nonfraud_idx)

X_rfe = X_train.loc[rfe_idx]
y_rfe = y_train.loc[rfe_idx]

print("\nRFE sample shape:", X_rfe.shape)

print("\nRFE sample class distribution:")
print(y_rfe.value_counts())


# --------------------------------------------------
# Configure and Run RFE
# --------------------------------------------------

rfe_estimator = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rfe = RFE(
    estimator=rfe_estimator,
    n_features_to_select=8,
    step=1
)

rfe.fit(
    X_rfe,
    y_rfe
)


# --------------------------------------------------
# Selected Features
# --------------------------------------------------

selected_features_rfe = (
    X_train.columns[rfe.support_].tolist()
)

print("\nNumber of selected features:",
      len(selected_features_rfe))

print("\nRFE Selected Features:")

for feature in selected_features_rfe:
    print("-", feature)


# --------------------------------------------------
# Prepare Reduced Dataset
# --------------------------------------------------

X_train_rfe = X_train[selected_features_rfe]
X_test_rfe = X_test[selected_features_rfe]

print("\nBaseline features:",
      X_train.shape[1])

print("RFE features:",
      X_train_rfe.shape[1])


# --------------------------------------------------
# Train RFE Random Forest
# --------------------------------------------------

rfe_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rfe_rf.fit(
    X_train_rfe,
    y_train
)

print("\nRFE Random Forest trained successfully.")
