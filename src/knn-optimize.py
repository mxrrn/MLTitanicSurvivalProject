'''
optimized kNN

optimizations for knn-optimize.py:
- double check preprocessing datasets -> holdout method
- feature subset selection - backward selection + cross validation
- fill missing values: optional - KNNImputer(n_neighbors=2, weights="uniform") - sklearn docu

optimizations for knn.py:
- hyperparameterize - for optimal k neighbors
- consider from sklearn.metrics import classification_report
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer  # KNN for missing value imputation
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocessing import preprocess_data  

# 1. load and input data
X_train, y_train = preprocess_data('data/train.csv', 'training')
X_test, y_test = preprocess_data('data/test.csv', 'testing')

# 2. resolve missing values - imputer
imputer = KNNImputer(n_neighbors=2, weights="uniform")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 4. Feature subset selection - backward selection + cross-validation
knn_base = KNeighborsClassifier(n_neighbors=5)
sfs = SequentialFeatureSelector(knn_base, n_features_to_select="auto", direction="backward", cv=5)

X_train_selected = sfs.fit_transform(X_train_scaled, y_train)
X_test_selected = sfs.transform(X_test_scaled)

selected_features = X_train.columns[sfs.get_support()].tolist()
print(f"Selected Features: {selected_features}")

# 5. Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_selected, y_train)

# 6. Predict
y_pred = knn.predict(X_test_selected)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'KNN Model Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)

# 8. Hyperparameter tuning
param_grid = {'n_neighbors': np.arange(1, 21)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_selected, y_train)

print(f'Best k: {grid_search.best_params_["n_neighbors"]}')
print(f'Best Accuracy: {grid_search.best_score_:.4f}')

# 9. Export results
output_file = "results/knn-optimize_results.txt"
with open(output_file, "w") as file:
    file.write(f"KNN Model Accuracy: {accuracy:.4f}\n")
    file.write("Confusion Matrix:\n")
    np.savetxt(file, conf_matrix, fmt="%d")  

    file.write("\nSelected Features:\n")
    file.write(", ".join(selected_features) + "\n")

    file.write(f"\nBest k (after tuning): {grid_search.best_params_['n_neighbors']}\n")
    file.write(f"Best Accuracy from Hyperparameter Tuning: {grid_search.best_score_:.4f}\n")

    file.write("\nGrid Search Results:\n")
    for param, mean_score in zip(grid_search.cv_results_["params"], grid_search.cv_results_["mean_test_score"]):
        file.write(f"{param} -> Mean Accuracy: {mean_score:.4f}\n")

print(f"Results saved to {output_file}")
