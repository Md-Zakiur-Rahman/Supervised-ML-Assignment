# Titanic Survival Prediction — CA-4

This project explores the Kaggle Titanic passenger dataset and compares five binary classifiers. The notebook contains the EDA, leakage-safe preprocessing, a stratified 70:30 holdout, training-only five-fold cross-validation, GridSearchCV for Random Forest and Gradient Boosting, class-weight comparison, ROC/AUC, confusion matrices and tree feature importance.

The Logistic Regression pipeline is retained as the **provisional** model because it led this holdout on accuracy, recall, F1 and ROC-AUC. The holdout results informed that choice, so they are no longer a fully independent final estimate. The project has not implemented the planned Flask demo.

## Project contents

```text
supervised/
├── app/                         # Reserved for the future Flask demo; currently empty
├── data/
│   ├── train.csv                # 891 labelled rows; used in the experiment
│   ├── test.csv                 # 418 rows without Survived; not used in training/evaluation
│   └── gender_submission.csv    # Kaggle sample predictions; not used in training/evaluation
├── models/
│   └── titanic_logistic_regression.joblib
├── notebooks/
│   └── titanic_ca4.ipynb
├── outputs/
│   ├── *.csv / *.json           # Metrics, CV, tuning, class-weight and importance results
│   └── graphs/                  # EDA, ROC, CV and feature-importance figures
├── reports/
│   ├── Titanic_Survival_Prediction_Report.docx
│   └── Titanic_Survival_Prediction_Report.pdf
├── README.md
└── requirements.txt
```

`test.csv` has 418 rows and no target column. `gender_submission.csv` contains `PassengerId` and `Survived` for 418 rows; its PassengerId values and row order match `test.csv`. These files are documented for completeness and are not part of the notebook experiment.

## Environment setup

The versions in `requirements.txt` match the environment used for the final QA pass (Python 3.14.4). Use the pinned scikit-learn and joblib versions when loading the serialized estimator.

```powershell
cd D:\jp_morgan\supervised
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

## Run the notebook

From the project root, launch Jupyter:

```powershell
jupyter notebook notebooks/titanic_ca4.ipynb
```

Run cells from top to bottom with the environment above selected as the notebook kernel. The notebook resolves data and output paths from either the project root or the `notebooks/` directory. It reads only `data/train.csv` for the existing experiment and writes graphs and result tables under `outputs/`.

## Fitted pipeline input contract

`models/titanic_logistic_regression.joblib` is the fitted baseline Logistic Regression `Pipeline` produced by the existing notebook configuration and 70:30 split (`random_state=42`). The prior notebook file retained outputs but not a serialized in-memory estimator, so the existing notebook was executed in a fresh kernel and `fitted_models["Logistic Regression"]` was saved without changing the source, model settings or evaluation. The saved result tables remain consistent. The pipeline expects one row per passenger with these columns:

| Column | Expected value |
|---|---|
| `Pclass` | Integer 1, 2 or 3; one-hot encoded as a category |
| `Sex` | `female` or `male` |
| `Age` | Numeric age; missing values are accepted and median-imputed |
| `Fare` | Numeric fare; missing values are accepted and median-imputed |
| `FamilySize` | Integer, calculated as `SibSp + Parch + 1` |
| `IsAlone` | Integer 0/1, derived from `FamilySize == 1` |
| `Embarked` | `C`, `Q` or `S`; missing values are accepted and mode-imputed |
| `Title` | `Mr`, `Mrs`, `Miss`, `Master` or `Rare`, extracted and normalized from `Name` |

Numeric columns are standardized; categorical columns are one-hot encoded, and unseen categories are ignored by the fitted encoder. If an eventual interface collects raw `SibSp`, `Parch` or `Name`, it must derive `FamilySize`, `IsAlone` and `Title` before calling the pipeline. No Flask route or user interface exists yet.

To regenerate the artifact after running the notebook through the baseline-model cell from the project root:

```python
import joblib
joblib.dump(
    fitted_models["Logistic Regression"],
    "models/titanic_logistic_regression.joblib",
    compress=3,
)
```

## Results and reports

- Executed notebook: `notebooks/titanic_ca4.ipynb`
- Metric, CV, class-weight, confusion-matrix and feature-importance tables: `outputs/`
- Figures: `outputs/graphs/`
- Academic report: `reports/Titanic_Survival_Prediction_Report.docx` and `.pdf`
- Fitted pipeline for later demo development: `models/titanic_logistic_regression.joblib`

The saved pipeline can be loaded with `joblib.load(...)`. Serialized scikit-learn models should be loaded with the pinned dependencies because cross-version compatibility is not guaranteed.

## Flask demo status

Flask is included in `requirements.txt` for the planned next stage. The `app/` directory is intentionally empty; no Flask application has been built.
