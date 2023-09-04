# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


area1, energy1 = energy_area_data("morphology_energy_112.xlsx") # morphology_energy_112.xlsx

area1 = np.array(area1)
energy1 = np.array(energy1)

X = np.array(area1)[1:]
y = np.array(energy1)[1:]

X = np.array(X, dtype=float)
y = np.array(y, dtype=float)

print('X shape', X.shape)
print('y shape', y.shape)

# Assuming you have already defined the dataset X and the target variable y

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the hyperparameter search space
param_grid = {
    'hidden_layer_sizes': [(50,), (30,), (20,), (50, 30), (50, 20), (30, 20), (50, 30, 20)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['adaptive'],
    'max_iter': [500],
    'tol': [0.0001]
}

# Create an MLPRegressor object
regressor = MLPRegressor()

# Create a GridSearchCV object
grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

# Fit the model using grid search
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the corresponding score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_

# Print the best hyperparameters and the corresponding score
print("Best Hyperparameters: ", best_params)
print("Best Score: ", best_score)

# Create a new MLPRegressor object with the best hyperparameters
best_regressor = MLPRegressor(**best_params)

# Fit the model using the best hyperparameters
best_regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = best_regressor.predict(X_test)

# Print the prediction results
print(y_pred)
