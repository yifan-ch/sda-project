"""
This module provides functions to calculate the Variance Inflation Factor (VIF)
for each predictor in a dataset, which helps in detecting multicollinearity.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_correlation_matrix(X):
    """
    Calculate the correlation matrix of the data.

    Parameters:
    X (numpy.ndarray): The input data matrix.

    Returns:
    numpy.ndarray: The correlation matrix.
    """

    n, m = X.shape
    correlation_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            X_i, X_j = X[:, i], X[:, j]
            mean_i, mean_j = np.mean(X_i), np.mean(X_j)
            covariance = np.sum((X_i - mean_i) * (X_j - mean_j)) / n
            std_i = np.sqrt(np.sum((X_i - mean_i) ** 2) / n)
            std_j = np.sqrt(np.sum((X_j - mean_j) ** 2) / n)
            correlation_matrix[i, j] = covariance / (std_i * std_j)
    return correlation_matrix


def standardize_data(X):
    """
    Standardize the data to have mean 0 and standard deviation 1.

    Parameters:
    X (numpy.ndarray): The input data matrix.

    Returns:
    numpy.ndarray: The standardized data matrix.
    """

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


def check_linear_dependency(X):
    """
    Check for linear dependency in the data.

    Parameters:
    X (numpy.ndarray): The input data matrix.

    Prints a warning if columns in X are linearly dependent or nearly so.
    """

    singular_values = np.linalg.svd(X, compute_uv=False)
    if np.min(singular_values) < 1e-10:
        print("Warning: Columns in X are linearly dependent or nearly so.")


def vif(df):
    """
    Calculate the variance inflation factor for each predictor.
    Also check linear dependency in the data.

    Parameters:
    df (pandas.DataFrame): The input data frame.

    Returns:
    pandas.DataFrame: DataFrame containing VIF values for each predictor.
    """

    check_linear_dependency(df.values)
    vif_values = calculate_vif(df)
    return vif_values


def calculate_vif(df):
    """
    Calculate the variance inflation factor for each predictor.

    Parameters:
    df (pandas.DataFrame): The input data frame.

    Returns:
    pandas.DataFrame: DataFrame containing VIF values for each predictor.
    """

    # Ensure data is standardized
    X = (df - df.mean()) / df.std()
    # Add a constant column for intercept in statsmodels
    df["intercept"] = 1

    # Compute VIF for each column
    vif = pd.DataFrame()
    vif["Predictor"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Drop the intercept VIF (not meaningful)
    return vif[vif["Predictor"] != "intercept"]
