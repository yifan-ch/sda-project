"""
This module contains functions for training and testing an ElasticNet regression model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV


def elastic_net_model(df):
    """
    Trains an ElasticNet regression model using
    the provided DataFrame and returns the reduced feature set.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing features and target variable.

    Returns:
    pd.DataFrame: The DataFrame containing only the selected features based on the trained model.
    """
    # Separating features and target variable
    X = df.drop("status", axis=1)
    y = df["status"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        "alpha": [0.05, 0.06, 0.07, 0.08, 0.1, 0.2, 0.3, 0.5],
        "l1_ratio": [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    }

    # Initialize ElasticNet model
    elastic_net = ElasticNet(random_state=42, max_iter=5000)

    # Perform grid search with cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, scoring="r2", cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Get feature importance (coefficients) from the best model
    coefficients = pd.Series(best_model.coef_, index=X.columns)

    # Identify features with non-zero coefficients
    # and filter the dataset to include only these features
    selected_features = coefficients[coefficients != 0].index
    X_reduced = X[selected_features]

    return X_reduced
