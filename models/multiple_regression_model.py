"""
Functions for training and testing a multiple linear regression model.
"""

import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from tools.model_tools import split_training_test, predict_class, calculate_metrics
from models.elastic_net_model import elastic_net_model


# from sklearn.metrics import (
#     mean_squared_error,
#     mean_absolute_error,
#     root_mean_squared_error,
#     r2_score,
# )


# def calculate_correlation_matrix_with_pandas(X, z_scores):
#     # Convert X back to DataFrame for easier handling if needed
#     X_df = pd.DataFrame(X, columns=z_scores.columns)
#     return X_df.corr()


def calculate_vif_with_statsmodels(X):
    vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return vif


def perform_regression_sklearn(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def perform_regression_statsmodels(X, y):
    # Add a constant term to the model (intercept)
    X_with_intercept = sm.add_constant(X)
    model = sm.OLS(y, X_with_intercept).fit()

    return model


def test_regression_sklearn(z_scores, random_state, frac_training=0.5, thres=0.5):
    # Split the data for status 0 (64 total samples)
    X_training, y_training, X_test, y_test = split_training_test(
        z_scores, random_state, frac_training
    )

    model = perform_regression_sklearn(X_training, y_training)

    # predict
    y_pred = predict_class(model.predict(X_test), thres)
    # mae = mean_absolute_error(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)
    # rmse = root_mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    metrics = calculate_metrics(y_test, y_pred)

    # return (mae, mse, rmse, r2), accuracy
    return metrics


def stats_mlr(df, frac_training=0.5, threshold=0.5, repetitions=100, use_elasticnet=False):
    """
    Perform multiple repetitions of the test and return the means
    """

    if use_elasticnet:
        z = elastic_net_model(df)
        z["status"] = df["status"]
        df = z

    metrics = np.mean(
        np.array(
            [
                np.array(test_regression_sklearn(df, random_state, frac_training, threshold))
                for random_state in range(repetitions)
            ]
        ),
        axis=0,
    )

    return metrics
