# Titanic Survival Prediction: k-NN (holdout version)

This branch holds my original, self-contained k-NN implementation for the Titanic survival task from a
university AI course. It was my part of a group project that compared several classifiers. The `main` branch
has the later version that shares a preprocessing pipeline with the other models.

## Approach

**`src/preprocessing.py`** prepares `data/train.csv`: it fills missing `Age` values with the median, turns
`Cabin` into category codes (missing cabins become "Unknown"), encodes `Sex` as 0/1, one-hot encodes
`Embarked` and drops `Name`, `Ticket` and `PassengerId`.

**`src/knn.py`** is the baseline:

1. Split the data into 80% training and 20% validation with a stratified holdout split (`random_state=42`),
   which gives a 179-passenger validation set.
2. Standardise the features. k-NN is distance-based, so unscaled features such as `Fare` would dominate.
   The scaler is fitted on the training part only.
3. Train a `KNeighborsClassifier` with k = 5 and evaluate it with accuracy and a confusion matrix.
4. Tune k from 1 to 20 with a 5-fold cross-validated grid search.

**`src/knn-optimize.py`** is the same pipeline plus two experiments: `KNNImputer` for missing values, and
forward feature selection (`SequentialFeatureSelector` with 5-fold cross-validation, which keeps half of the
features with the `"auto"` setting). **In this version the imputed data and the selected features are computed
but not passed on to the model**, so the model is still trained and tuned on all features and the results are
identical to the baseline. The `main` branch applies the feature selection properly.

## Results

| Script | Validation accuracy (k = 5) | Best k (5-fold CV) | CV accuracy |
|---|---|---|---|
| `knn.py` | 0.7877 | 7 | 0.8217 |
| `knn-optimize.py` | 0.7877 | 7 | 0.8217 |

Confusion matrix on the validation set (rows: actual, columns: predicted):

| | Predicted died | Predicted survived |
|---|---|---|
| **Died** | 95 | 15 |
| **Survived** | 23 | 46 |

The full output is in `results/`. The final model uses k = 5, and the tuned k is only reported, not used for
the validation score. With only 179 validation passengers, a difference of a few passengers moves the accuracy
by about 0.5 percentage points, so small differences between settings should not be over-interpreted.

## Setup and usage

```
pip install -r requirements.txt
python src/knn.py
python src/knn-optimize.py
```

Run both from the project root. They print their progress and write a results file into `results/`.

## Project structure

```
.
├── data/                  train.csv and test.csv (test.csv is loaded but not used for evaluation)
├── src/                   preprocessing and the two k-NN scripts
├── results/               output of the scripts
└── requirements.txt
```

## Tech

Python, pandas, scikit-learn, NumPy.
