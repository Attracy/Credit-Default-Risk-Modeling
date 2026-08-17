# Credit Default Risk Modeling

A supervised classification project predicting serious credit delinquency within 2 years, built on the "Give Me Some Credit" dataset (150K+ records), comparing Logistic Regression, Decision Tree, Random Forest, and XGBoost with careful, validation-driven outlier and feature treatment.

---

## Objective

Predict `SeriousDlqin2yrs` — whether a borrower will experience serious delinquency (90+ days past due) within the next two years — and identify which financial behaviors drive that risk.

## Dataset

- Give Me Some Credit (Kaggle), ~150,000 borrower records
- Target: `SeriousDlqin2yrs` (binary) — default rate 6.953%, consistent across train/test after stratified split
- Features: revolving utilization, age, past-due counts (30-59, 60-89, 90+ days), debt ratio, monthly income, open credit lines, real estate loans, dependents

## Methodology

The core discipline of this project: every cleaning and feature-engineering decision is validated against a 5-fold cross-validated AUC baseline (`validate()` helper, Logistic Regression + Decision Tree) before being accepted — nothing is applied on assumption alone.

### Step 3 — Train-Test Split
80/20 stratified split on the target, preserving the 6.953% default rate in both sets, so no distributional shift is introduced before modeling begins.

### Step 4 — Outlier Detection & Treatment
Each treatment below is validated before/after with the `validate()` baseline check:

- Age: rows with `age == 0` (a handful) dropped from train; equivalent test rows imputed with train minimum rather than dropped, to keep the test set intact.
- Delinquency columns (`NumberOfTime30-59...`, `60-89...`, `NumberOfTimes90DaysLate`): sentinel codes 96/98 (data-entry artifacts, not real counts) flagged into a new binary feature `is_coded`, then the three columns are capped at their max value below 90 to remove the coding artifacts.
- Revolving Utilization: capped at the 99th percentile after checking that extreme values above 1–3x still carried real default signal, not just noise.
- Debt Ratio: heavily right-skewed; several percentile cutoffs (50th to 99th) were checked against actual default rates before deciding on a treatment. Ultimately quantile-binned into 10 ordinal bins (`KBinsDiscretizer`) rather than capped, since raw capping didn't preserve the same signal.
- Monthly Income: near-zero incomes (≤ 1) flagged into a `low_income_flag` binary feature before capping, so the "no reported income" signal isn't lost in the cap; then capped at the 99th percentile.

### Step 5 — Feature Engineering
`MonthlyIncome` log-transformed (`log1p`) to reduce right-skew after capping — validated to hold or improve AUC before finalizing.

### Step 6 — Feature Selection
Multicollinearity checked via Variance Inflation Factor (VIF) on standardized features. All VIF values came in under 3.3 (`is_coded` highest at 3.30) — no multicollinearity severe enough to warrant dropping features.

### Step 7 — Model Building
Four models tuned via `GridSearchCV(cv=5, scoring='roc_auc')` on the same stratified folds:

| Model | Best CV AUC | Best Params |
|---|---|---|
| Logistic Regression | 0.8500 | C=10 |
| Decision Tree | 0.8449 | max_depth=8, min_samples_leaf=200 |
| Random Forest | 0.8561 | n_estimators=500, max_depth=14, min_samples_leaf=50 |
| XGBoost | 0.8576 | n_estimators=300, max_depth=3, learning_rate=0.05 |

Class imbalance handled per-model: `class_weight='balanced'` for Logistic Regression/Decision Tree/Random Forest, `scale_pos_weight` (ratio of negatives to positives, computed from train only) for XGBoost.

Classification threshold per model is not the default 0.5 — it's selected from cross-validated predictions to maximize F-beta (beta=2, recall weighted twice precision, reflecting that missing a defaulter is costlier than a false alarm). Test-set thresholds are carried over from train CV, never refit on test.

## Results

Test-set performance (test set used exactly once, thresholds fixed from train CV):

| Model | AUC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.8390 | 0.2567 | 0.6415 | 0.3667 | 0.8459 |
| Decision Tree | 0.8334 | 0.2865 | 0.5990 | 0.3876 | 0.8684 |
| Random Forest | 0.8437 | 0.2639 | 0.6403 | 0.3738 | 0.8508 |
| XGBoost (best) | 0.8444 | 0.2781 | 0.6314 | 0.3861 | 0.8604 |

XGBoost is selected as the final model — highest test AUC and a strong balance of recall (63%) against precision, catching roughly 2 in 3 actual defaulters while keeping accuracy above 86%.

XGBoost confusion matrix (threshold ≈ 0.606):

```
                Predicted 0   Predicted 1
Actual 0          19,624         2,739
Actual 1             616         1,055
```

## Feature Importance

Consistent across all four models, the strongest predictors of default are:

1. Revolving Utilization of Unsecured Lines — top feature by a wide margin in every tree-based model
2. NumberOfTimes90DaysLate
3. NumberOfTime30-59DaysPastDueNotWorse
4. NumberOfTime60-89DaysPastDueNotWorse
5. The engineered `is_coded` flag (Logistic Regression's top feature — the sentinel-code artifact itself carries predictive signal)

Past-due history and credit utilization dominate over income, dependents, and real estate loans across every model — consistent with standard credit-risk intuition.

## Tech Stack

- Data processing: pandas, NumPy
- Modeling: scikit-learn (Logistic Regression, Decision Tree, Random Forest, GridSearchCV, StratifiedKFold), XGBoost
- Diagnostics: statsmodels (VIF)
- Visualization: matplotlib, seaborn

## Repository Structure

```
├── Step_3_onwards_with_models_current.ipynb   # Train-test split through final model comparison
├── step3_clean.csv                             # Cleaned input from earlier EDA steps
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost statsmodels matplotlib seaborn
jupyter notebook Step_3_onwards_with_models_current.ipynb
```

Run all cells sequentially — each outlier/feature-engineering step is validated in place before the next is applied, so order matters.

## Author

Shashank Shekhar — M.Sc. Statistics (Applied Statistics and Informatics), IIT Bombay
