PaySim Fraud Detection Using Feature Engineering and Random Forest

## Project Overview

This project applies the CRISP-DM methodology to financial fraud detection using the PaySim mobile-money transaction dataset. A Random Forest classifier is used as the baseline model, and Recursive Feature Elimination (RFE) is applied to identify a smaller set of informative predictors.

The project compares the predictive performance of the baseline Random Forest with an RFE-based Random Forest while maintaining a reproducible workflow for data preparation, modeling, evaluation, and deployment.

### Research Questions

1. Which transaction characteristics are the most important predictors of fraudulent financial transactions?
2. Can Recursive Feature Elimination reduce the number of predictors while maintaining strong Random Forest fraud-detection performance?
Dataset Information

PaySim contains 6,362,620 simulated transactions and 11 variables. The target is isFraud. Only 8,213 transactions (about 0.13%) are fraudulent, creating severe class imbalance.

Expected file:

'data/raw/Fraud_Detection_Dataset.csv'

GitHub note: the full CSV exceeds GitHub's normal 100 MB single-file limit. For a public GitHub repository, use Git LFS. A public Google Drive/Colab folder is also acceptable if your instructor permits it.

CRISP-DM Process

Business Understanding — define fraud problem, objectives, RQs, and metrics.

Data Understanding — inspect structure, quality, distributions, transaction types, and fraud patterns.

Data Preparation — remove identifiers, encode transaction type, engineer behavioral features, and split data.

 **Modeling** — train a baseline Random Forest, apply Recursive Feature Elimination (RFE), and train a second Random Forest using the selected features.

 **Evaluation** — compare the baseline and RFE Random Forest models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

**Deployment** — save the trained RFE Random Forest model and selected features so they can be loaded for future transaction scoring.

## Repository Structure

``` text
paysim-fraud-detection-crispdm/
│
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
│
├── notebooks/
│   ├── DDS-7255 Assignment 5 (1).ipynb
│   └── README.md
│
├── outputs/
│   └── README.md
│
├── scripts/
│   ├── 01_business_understanding.py
│   ├── 02_data_understanding.py
│   ├── 03_data_preparation.py
│   ├── 04_modeling.py
│   ├── 05_evaluation.py
│   └── 06_deployment.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run the Project

Clone the repository:

```bash
git clone <repository-url>
cd paysim-fraud-detection-crispdm
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the PaySim dataset in:

```text
data/raw/Fraud_Detection_Dataset.csv
```

Then run the CRISP-DM scripts in order:

```bash
python scripts/01_business_understanding.py
python scripts/02_data_understanding.py
python scripts/03_data_preparation.py
python scripts/04_modeling.py
python scripts/05_evaluation.py
python scripts/06_deployment.py
```

You may also run the original analysis notebook located in the `notebooks/` directory.

## Dependencies

Python 3.10+, pandas, NumPy, Matplotlib, scikit-learn, imbalanced-learn, joblib, and Jupyter.


## Results and Insights

The reproducible workflow compares two Random Forest configurations:

- **Baseline Random Forest:** trained using the complete prepared feature set.
- **RFE Random Forest:** trained using the eight predictors selected through Recursive Feature Elimination.

Model performance is evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices. Running `scripts/05_evaluation.py` generates the final comparison and saves it as:

`outputs/evaluation_metrics.csv`

RFE reduces the dimensionality of the model from the full predictor set to eight selected features, allowing the project to evaluate whether a more compact feature set can maintain strong fraud-detection performance.

The selected predictors are saved to:

`outputs/selected_features.txt`

The four Random Forest configurations produced the following results:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 99.97% | 97.65% | 78.39% | 86.97% | 99.38% |
| PCA | 99.93% | 95.26% | 45.28% | 61.39% | 90.22% |
| Engineered Features | 99.9996% | 99.94% | 99.76% | 99.85% | 99.88% |
| Engineered + SMOTE | 99.9990% | 99.45% | 99.76% | 99.61% | 99.91% |

The engineered-feature model achieved the strongest overall balance of precision, recall, and F1-score. PCA substantially reduced fraud-detection performance, particularly recall. SMOTE maintained very high recall but provided only a marginal improvement after strong domain-driven features had already been created.
