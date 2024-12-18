"""
Functions for training and testing a ElasticNet regression model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV


def elastic_net_model(df):
    # Separating features and target variable
    X = df.drop("status", axis=1)
    y = df["status"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

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

    best_model = grid_search.best_estimator_

    # Get feature importance
    coefficients = pd.Series(best_model.coef_, index=X.columns)

    # Identify features and filter the dataset to include only selected features
    selected_features = coefficients[coefficients != 0].index
    X_reduced = X[selected_features]

    return X_reduced


if __name__ == "_main_":
    elastic_net_model()
