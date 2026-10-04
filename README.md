<div align="center">
  <h3>Titanic Survival Prediction</h3>
  <p>Machine Learning in Python — Mini Project 3</p>
</div>

Predicting passenger survival on the [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic).

Six models compared. Best result: **XGBoost at 83.24%** cross-validated accuracy,
against a 61.75% baseline.

---

### Models

| Rank | Model | CV Accuracy | Std |
|---:|---|---:|---:|
| 1 | **XGBoost** | **83.24%** | 1.85 |
| 2 | Logistic Regression | 82.68% | 1.66 |
| 3 | Random Forest | 80.31% | 2.12 |
| 4 | Decision Tree | 77.27% | 3.41 |
| 5 | KNN | 70.75% | 1.57 |
| 6 | SVM | 68.51% | 1.83 |

Five-fold stratified cross-validation, `random_state=42`, so every model saw
identical splits and the numbers are directly comparable.

### What drove survival

`Sex` 74% vs 19% · `Pclass` 63% first class vs 24% third · `Title` children
survived far better than adults · passengers with family outperformed the
alone

### Pipeline

`FamilySize` and `IsAlone` · `Title` extracted from `Name` · median imputation ·
one-hot and label encoding · age and family binning · train-test split and
feature scaling · `GridSearchCV` tuning on the winner

### Files

`main.ipynb` — notebook · `train.csv` / `test.csv` — data ·
`submission.csv` — predictions · `model_comparison.csv` — score table

### Submit

```bash
kaggle competitions submit -c titanic -f submission.csv
```

### Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter nbconvert --to notebook --execute main.ipynb
```