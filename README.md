# Titanic Survival Prediction

Machine Learning in Python — Mini Project 3

Predict whether a passenger survived the Titanic sinking, using the
[Kaggle Titanic dataset](https://www.kaggle.com/c/titanic).

## Result

| Model | CV Accuracy | CV Std | Single Split |
|---|---|---|---|
| **XGBoost** | **83.24%** | 1.85 | 84.27% |
| Logistic Regression | 82.68% | 1.66 | 81.46% |
| Random Forest | 80.31% | 2.12 | 78.65% |
| Decision Tree | 77.27% | 3.41 | 81.46% |
| KNN | 70.75% | 1.57 | 77.53% |
| SVM | 68.51% | 1.83 | 81.46% |

Baseline (always predicting the majority class): **61.75%**

Cross-validated with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

## What the model learned

The strongest signals in the data:

- **Sex** — 74% of women survived, only 19% of men
- **Pclass** — 63% survival in first class, 24% in third class
- **Title** — extracted from `Name`; `Master.` marks a young boy, and
  children had priority seats in the lifeboats
- **Age** — children survived at a much higher rate than adults
- **Family** — passengers travelling with family did better than those
  travelling alone

## Approach

1. **EDA** — shape, null counts, target balance, distributions
2. **Visualisation** — histograms, correlation heatmap, survival-rate bars
3. **Preprocessing** — `FamilySize`, `IsAlone`, `Title` extraction,
   median imputation, one-hot and label encoding, age and family binning
4. **Modeling** — six algorithms compared on identical splits
5. **Evaluation** — 5-fold cross-validation plus `GridSearchCV` tuning
6. **Submission** — `submission.csv` in the Kaggle format

## Files

| File | Description |
|---|---|
| `main.ipynb` | The full notebook |
| `train.csv` | Training data with the `Survived` label (891 rows) |
| `test.csv` | Competition data, no label (418 rows) |
| `sampleSubmission.csv` | Expected output format from Kaggle |
| `submission.csv` | My predictions for `test.csv` |
| `model_comparison.csv` | Accuracy table for all six models |
| `assets/alwer.jpeg` | Logo |

## Submitting to Kaggle

```bash
kaggle competitions submit -c titanic -f submission.csv
```

## Running locally

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter nbconvert --to notebook --execute main.ipynb
```