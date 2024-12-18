"""
This module provides functions to perform multiple linear regression using
both statsmodels and scikit-learn, and to evaluate the model performance.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from tools.model_tools import split_training_test, predict_class, calculate_metrics
from models.elastic_net_model import elastic_net_model


def perform_regression_statsmodels(X, y):
    """
    Perform multiple linear regression using statsmodels.

    Parameters:
    X (numpy.ndarray): The input feature matrix.
    y (numpy.ndarray): The target variable.

    Returns:
    statsmodels.regression.linear_model.RegressionResultsWrapper: The fitted model.
    """

    # Add a constant term to the model (intercept)
    X_with_intercept = sm.add_constant(X)
    model = sm.OLS(y, X_with_intercept).fit()

    return model


def perform_regression(X, y):
    """
    Perform multiple linear regression using scikit-learn.

    Parameters:
    X (numpy.ndarray): The input feature matrix.
    y (numpy.ndarray): The target variable.

    Returns:
    sklearn.linear_model.LinearRegression: The fitted model.
    """

    model = LinearRegression()
    model.fit(X, y)
    return model


def test_regression(df, random_state, frac_training=0.5, threshold=0.5):
    """
    Train and test a multiple linear regression model.

    Parameters:
    df (pandas.DataFrame): The input data frame.
    random_state (int): The random state for data splitting.
    frac_training (float): The fraction of data to be used for training.
    threshold (float): The threshold for class prediction.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """

    X_training, y_training, X_test, y_test = split_training_test(df, random_state, frac_training)

    model = perform_regression(X_training, y_training)
    y_pred = predict_class(model.predict(X_test), threshold)
    metrics = calculate_metrics(y_test, y_pred)

    return metrics


def stats_mlr(df, frac_training=0.5, threshold=0.5, repetitions=100, use_elasticnet=False):
    """
    Perform multiple repetitions of the test and return the means.

    Parameters:
    df (pandas.DataFrame): The input data frame.
    frac_training (float): The fraction of data to be used for training.
    threshold (float): The threshold for class prediction.
    repetitions (int): The number of repetitions for the test.
    use_elasticnet (bool): Whether to use elastic net regularization.

    Returns:
    numpy.ndarray: The mean evaluation metrics across repetitions.
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
