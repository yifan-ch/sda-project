"""
Functions for training and testing a multiple linear regression model.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from tools.model_tools import split_training_test, predict_class, calculate_metrics
from models.elastic_net_model import elastic_net_model


def perform_regression_statsmodels(X, y):
    """
    Perform multiple linear regression using statsmodels.
    """

    # Add a constant term to the model (intercept)
    X_with_intercept = sm.add_constant(X)
    model = sm.OLS(y, X_with_intercept).fit()

    return model


def perform_regression(X, y):
    """
    Perform multiple linear regression using scikit-learn.
    """

    model = LinearRegression()
    model.fit(X, y)
    return model


def test_regression(df, random_state, frac_training=0.5, threshold=0.5):
    """
    Train and test a multiple linear regression model.
    """

    X_training, y_training, X_test, y_test = split_training_test(df, random_state, frac_training)

    model = perform_regression(X_training, y_training)
    y_pred = predict_class(model.predict(X_test), threshold)
    metrics = calculate_metrics(y_test, y_pred)

    return metrics


def stats_mlr(df, frac_training=0.5, threshold=0.5, repetitions=100, use_elasticnet=False):
    """
    Perform multiple repetitions of the test and return the means
    """

    # Perform elastic net if wanted
    if use_elasticnet:
        z = elastic_net_model(df)
        z["status"] = df["status"]
        df = z

    # Perform multiple linear regression for wanted repetitions
    metrics = np.mean(
        np.array(
            [
                np.array(test_regression(df, random_state, frac_training, threshold))
                for random_state in range(repetitions)
            ]
        ),
        axis=0,
    )
    return metrics
