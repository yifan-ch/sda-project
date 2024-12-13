"""
Functions for training and testing a ElasticNet regression model.
"""

# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


def elastic_net_model():
    data = pd.read_csv("data/generated/parkinsons_z_scores.csv")

    # Separating features and target variable
    X = data.drop("status", axis=1)
    y = data["status"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    # Initialize Elastic Net
    # elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)

    # Train the model
    # elastic_net.fit(X_train, y_train)

    # Predict on test set
    # y_pred = elastic_net.predict(X_test)

    # Calculate performance metrics
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # print(f"Mean Squared Error: {mse}")
    # print(f"R-squared: {r2}")

    # Define parameter grid
    param_grid = {
        "alpha": [0.05, 0.06, 0.07, 0.08, 0.1, 0.2, 0.3, 0.5],
        "l1_ratio": [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    }

    # Initialize elastic net model
    elastic_net = ElasticNet(random_state=42, max_iter=5000)

    # Grid search with cross-validation
    grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, scoring="r2", cv=5)
    grid_search.fit(X_train, y_train)

    # Best parameters
    print("Best parameters:", grid_search.best_params_)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate best model on test set
    y_pred_best = best_model.predict(X_test)
    mse_best = mean_squared_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)

    print(f"Optimized Mean Squared Error: {mse_best}")
    print(f"Optimized R-squared: {r2_best}")

    # Get feature importance
    coefficients = pd.Series(best_model.coef_, index=X.columns)

    # Identify features
    selected_features = coefficients[coefficients != 0].index
    print("Selected features:", selected_features)

    # Filter the dataset to include only selected features
    X_reduced = X[selected_features]
    return X_reduced


if __name__ == "_main_":
    elastic_net_model()
