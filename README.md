# LD6051 - Predicting Student Academic Performance Using Cloud-Based Machine Learning

Group project for **LD6051 Machine Learning on the Cloud**.
Built and run on **Google Colaboratory** (free CPU runtime).

[![Click here To Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev-Panchal1/LD6051_ML_Group_Project_Student-Performance/blob/main/LD6051_ML_Group_Project.ipynb)

---

## Project overview

This project applies machine learning to predict student academic performance
and to identify which factors most influence exam results. Three classifiers are
trained and compared on the same data, and the strongest model is identified.

| Item | Detail |
|---|---|
| **Cloud platform** | Google Colaboratory |
| **Dataset** | `StudentPerformanceFactors.csv` - 6,607 records, 19 input features |
| **Dataset source** | [Kaggle - Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) |
| **Task** | Multi-class classification (Low / Medium / High performance) |
| **Best model** | Support Vector Machine (RBF) - 85.70% accuracy, macro-F1 0.85 |

---

## Repository contents

| File | Description |
|---|---|
| `LD6051_ML_Group_Project.ipynb` | The complete notebook - data loading through to model comparison, with all outputs and charts saved |
| `StudentPerformanceFactors.csv` | The dataset used |
| `README.md` | This file |

---

## How to run

**Option 1 - Google Colab (Recommended)**

1. Click the *Open in Colab* badge above.
2. Select **Runtime - Run all**.
3. When prompted, upload `StudentPerformanceFactors.csv` (also in this repository).

**Option 2 - Locally with Jupyter**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook LD6051_ML_Group_Project.ipynb
```

Place `StudentPerformanceFactors.csv` in the same folder. The loading cell
detects that it is not running in Colab and reads the file directly.

A fixed seed (`random_state = 42`) is used throughout, so re-running the
notebook reproduces every figure below exactly.

---

## Method

1. **Exploratory data analysis** - distributions, boxplots by category and a
   correlation heatmap.
2. **Feature engineering** - the continuous `Exam_Score` is converted into three
   ordinal classes using quantile binning (`pd.qcut`), because the raw score has
   a standard deviation of only 3.89 marks and an interquartile range of four
   marks. `Exam_Score` is then dropped to prevent target leakage.
3. **Pre-processing** - missing values retained as a distinct category, all 13
   categorical columns integer-encoded, `StandardScaler` fitted on the training
   partition only.
4. **Split** - stratified 80/20 (5,285 training / 1,322 test records).
5. **Models** - Decision Tree, Random Forest and SVM (RBF), all evaluated on the
   identical test partition.

### Target classes

| Class Code | Label | Exam Score Range | Records |
|---|---|---|---|
| 0 | High | 70 - 101 | 1,625 (24.6%) |
| 1 | Low | 55 - 66 | 2,882 (43.6%) |
| 2 | Medium | 67 - 69 | 2,100 (31.8%) |

> `LabelEncoder` assigns codes alphabetically, which is why High = 0 rather than
> the intuitive ordering. This mapping is needed to read the confusion matrices
> correctly.

---

## Results

| Model | Accuracy | Macro-F1 | F1 High | F1 Low | F1 Medium |
|---|---|---|---|---|---|
| Decision Tree (`max_depth=5`) | 68.91% | 0.68 | 0.72 | 0.79 | 0.54 |
| Random Forest (200 trees, depth 10) | 79.20% | 0.78 | 0.81 | 0.86 | 0.67 |
| **SVM (RBF, C = 1.0)** | **85.70%** | **0.85** | **0.86** | **0.91** | **0.79** |
| Majority-class baseline | 43.62% | 0.20 | 0.00 | 0.61 | 0.00 |

All three models comfortably beat the majority-class baseline, confirming that
each learned genuine structure in the data.

### Most important factors

`Attendance` (0.381), `Hours_Studied` (0.220) and `Previous_Scores` (0.080)
together account for roughly 68% of the total feature importance. Self-reported
`Motivation_Level` contributes very little. No demographic variable appears in
the top three.

---

## Known Limitations

1. `LabelEncoder` assigns codes alphabetically, scrambling the true
   Low < Medium < High ordering. Harmless for the tree models, but a genuine
   weakness for the SVM's distance-based kernel.
2. No hyperparameter search - the settings are reasoned defaults, so these are
   baseline rather than optimised results.
3. Evaluation rests on a single 80/20 split rather than k-fold cross-validation.
4. Class imbalance was managed (stratification, macro-F1) rather than corrected.
5. Demographic features were retained so the model *could* be audited for bias,
   but no fairness audit was performed.

## Recommended next steps

- Replace `LabelEncoder` with `OrdinalEncoder` using declared category orders.
- Add stratified k-fold cross-validation.
- Tune hyperparameters with `GridSearchCV`.
- Test `class_weight="balanced"`.
- Run a per-subgroup fairness audit and add explainability (e.g. SHAP).

---

## Ethical Note

This model should support human judgement, not replace it. Even at 85.70%
accuracy, roughly one prediction in seven is incorrect, and the error rate is
higher still for the Medium band. A prediction is not a judgement about a
student.
