# Titanic Survival Prediction: k-Nearest Neighbours

Predicting who survived the Titanic from passenger data, a tabular classification task from a university AI
course. The course project was done in a group that compared several classifiers. **My contribution was the
k-NN classifier**: implementation, feature selection, hyperparameter tuning and evaluation. This repository
contains that part together with the preprocessing it builds on.

## Task and data

The data is the classic Kaggle Titanic dataset, split into `data/train.csv` (891 passengers with the label
`Survived`) and `data/test.csv` (418 passengers, also labelled). The goal is to predict `Survived` (0/1) from
class, sex, age, family members aboard, ticket, fare, cabin and port of embarkation.

## Pipeline

**Preprocessing** (`src/preprocessing.py`, `src/age_modelling.py`):

- `Sex` is encoded as 0/1, `Pclass`, `Sex` and `Embarked` are one-hot encoded.
- Missing `Fare` is filled with the median and missing `Embarked` with the most common port.
- Missing `Age` is filled by `model_ages` with the median age of the passenger's class and sex.
- `Cabin` becomes a category code, and missing cabins are one category ("Unknown").
- Derived features: `TicketFrequency` (people sharing a ticket), `IsAloneTicket`, and a grouped family size
  (alone, small, medium, large).
- `Name`, `Ticket`, `PassengerId` and the raw family size are dropped, leaving 19 features.

**Baseline k-NN** (`src/knn.py`):

1. Standardise all features. k-NN is distance-based, so unscaled features such as `Fare` would dominate.
   The scaler is fitted on the training data only.
2. Train a `KNeighborsClassifier` with k = 5 and evaluate it on the test set (accuracy and confusion matrix).
3. Tune k from 1 to 20 with a 5-fold cross-validated grid search on the training data.

**Optimised k-NN** (`src/knn-optimize.py`), on top of the baseline:

1. `KNNImputer` (2 neighbours) fills any remaining missing values.
2. `SequentialFeatureSelector` with backward elimination and 5-fold cross-validation removes features that
   do not help. With `n_features_to_select="auto"` and no tolerance, scikit-learn keeps about half of them.
3. Train, evaluate and tune k as before, now on the selected features.

## Results

| | Features | Test accuracy (k = 5) | Best k (5-fold CV) | CV accuracy at best k |
|---|---|---|---|---|
| `knn.py` (baseline) | all 19 | 0.7440 | 3 | 0.8014 |
| `knn-optimize.py` | 10 selected | 0.7392 | 5 | 0.8305 |

Selected features: `Age`, `Cabin`, `Sex_0`, `Sex_1`, `Pclass_3`, `Embarked_C`, `TicketFrequency`,
`IsAloneTicket`, `Family_Size_Grouped_Alone`, `Family_Size_Grouped_Medium`.

Confusion matrices on the test set (rows: actual, columns: predicted):

| Baseline | died | survived | Optimised | died | survived |
|---|---|---|---|---|---|
| **died** | 205 | 58 | **died** | 196 | 67 |
| **survived** | 49 | 106 | **survived** | 42 | 113 |

How to read this:

- Feature selection raises the cross-validated accuracy on the training data from about 0.80 to 0.83, but
  the test accuracy stays at about 0.74. The optimised model recovers more survivors (113 instead of 106)
  at the cost of more false alarms.
- The gap between cross-validation (0.80 to 0.83) and test accuracy (0.74) is large. The features and k were
  chosen on the same cross-validation folds they are scored on, so the cross-validation numbers are
  optimistic, and the test set may also be distributed differently from the training set.
- The test accuracy uses k = 5 in both scripts. For the optimised model this is also the tuned value.

The grid search tables are in `results/knn_results.txt` and `results/knn-optimize_results.txt`.

## Setup and usage

Python 3.10 or newer is recommended.

```
pip install -r requirements.txt
python src/knn.py
python src/knn-optimize.py
```

Run the scripts from the project root, since the data paths are relative (`data/train.csv`). They print their
progress and write the results into `results/`.

## Project structure

```
.
├── data/                  train.csv and test.csv
├── src/
│   ├── preprocessing.py   Shared feature engineering
│   ├── age_modelling.py   Age imputation
│   ├── knn.py             Baseline k-NN
│   └── knn-optimize.py    k-NN with imputation and feature selection
├── results/               Output of the scripts
└── requirements.txt
```

## Notes and limitations

- `age_modelling.py` replaces the group's original age imputation, which was not available when this
  repository was put together. Results can differ slightly from those of the original course submission.
- `Cabin` is encoded with category codes computed separately for the training and test data, so the same cabin
  can get different codes in the two sets, and the integer codes have no meaningful distance for k-NN. Dropping
  the feature or using the deck letter would be cleaner.
- The `knn-holdout` branch contains my earlier version, which evaluates on a holdout split of the training
  data and has its own simple preprocessing.

## Tech

Python, pandas, scikit-learn, NumPy.
