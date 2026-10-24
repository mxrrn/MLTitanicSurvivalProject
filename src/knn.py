'''
Merna: basic kNN
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# zum Analysieren des Ergebnisses: wie viel korrekt sind, modelleistung evaluieren
from sklearn.metrics import accuracy_score, confusion_matrix

# zusätzliche modelleistund metriken: precision, recall, F1-score
# from sklearn.metrics import classification_report

# folgendes importiert den preprocessing funktion - um preprocessing logic wiederzuverwenden
from preprocessing import preprocess_data  

'''
optimizations for knn-optimize.py:
- double check preprocessing datasets -> holdout method
- feature subset selection - forward then cross validation
- fill missing values: optional - KNNImputer(n_neighbors=2, weights="uniform") - sklearn docu

optimizations for knn.py:
- hyperparameterize - for optimal k neighbors
- consider from sklearn.metrics import classification_report

'''

# 1. Daten laden und preprocessen
# die Dateien sind von unsere preprocessing.py geladen 
train_data = preprocess_data('data/train.csv', 'data/test.csv')


# 2. Zerlegen der Daten: 
# X = input (Features) - außer Survived
# y = target ("Survived", also ob eine Person überlebt hat oder nicht)
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# 3. Aufteilen des Datensatzes - Holdout method:
# 80% für das Training (`X_train`, `y_train`)
# 20% für die Validierung (`X_test`, `y_test`)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 4. Feature Scaling (Normierung der Daten)
# KNN basiert auf Distanzen zwischen Datenpunkten 
# scaling parameters wird vom training daten gelernt und dann angewendet
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)

# 5. KNN initialisieren - auf training daten gefittet
knn = KNeighborsClassifier(n_neighbors=5)  # k-value k=5
knn.fit(X_train_scaled, y_train)

# 6. Predict anhand trainierte model
y_pred = knn.predict(X_test_scaled)

# 7. Modelleistung evaluieren
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

# Ergebnisse als text Datei exportieren
output_file = "results/knn_results.txt"
with open(output_file, "w") as file:
    file.write(f"KNN Model Accuracy: {accuracy:.4f}\n")
    file.write("Confusion Matrix:\n")
    np.savetxt(file, conf_matrix, fmt="%d")  

    file.write(f"\nBest k (after tuning): {grid_search.best_params_['n_neighbors']}\n")
    file.write(f"Best Accuracy from Hyperparameter Tuning: {grid_search.best_score_:.4f}\n")

print(f"Results saved to {output_file}")