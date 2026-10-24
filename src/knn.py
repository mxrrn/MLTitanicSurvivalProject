'''
basic kNN
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# to analyze the results and see how much is accurate and correct
from sklearn.metrics import accuracy_score, confusion_matrix


# import our own preprocessing function
from preprocessing import preprocess_data  

'''
optimizations for knn-optimize.py:
- double check preprocessing datasets -> holdout method
- feature subset selection - forward then cross validation / backward
- find optimal feature selection
- fill missing values: optional - KNNImputer(n_neighbors=2, weights="uniform") - sklearn docu

optimizations for knn.py:
- hyperparameterize - for optimal k neighbors
- consider from sklearn.metrics import classification_report

'''


# 1. load and input data
X_train, y_train = preprocess_data('data/train.csv', 'training')
X_test, y_test = preprocess_data('data/test.csv', 'testing')

# 2. feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)


# 3. initiate knn and fit the data
# include knn classifier 1 as well to demonstrate relevance of k value
knn = KNeighborsClassifier(n_neighbors=5)  # k-value k=5
knn.fit(X_train_scaled, y_train)

# 4. run prediction on test -- generate results
y_pred = knn.predict(X_test_scaled)


# 5. evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'KNN Model Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)

# Hyperparameter tuning -- optimal k
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': np.arange(1, 21)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f'Best k: {grid_search.best_params_["n_neighbors"]}')
print(f'Best Accuracy: {grid_search.best_score_:.4f}')

# export evaluation of the model
output_file = "results/knn_results.txt"
with open(output_file, "w") as file:
    file.write(f"KNN Model Accuracy: {accuracy:.4f}\n")
    file.write("Confusion Matrix:\n")
    np.savetxt(file, conf_matrix, fmt="%d")  

    file.write("\nSelected Features:\n")
    file.write(", ".join(X_train.columns) + "\n")


    file.write(f"\nBest k (after tuning): {grid_search.best_params_['n_neighbors']}\n")
    file.write(f"Best Accuracy from Hyperparameter Tuning: {grid_search.best_score_:.4f}\n")

    # Save all hyperparameter tuning results
    file.write("\nGrid Search Results:\n")
    for param, mean_score in zip(grid_search.cv_results_["params"], grid_search.cv_results_["mean_test_score"]):
        file.write(f"{param} -> Mean Accuracy: {mean_score:.4f}\n")

print(f"Results saved to {output_file}")



