import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_correlation_matrix(X):
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
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


def check_linear_dependency(X):
    singular_values = np.linalg.svd(X, compute_uv=False)
    # print("Singular values:", singular_values)
    if np.min(singular_values) < 1e-10:
        print("Warning: Columns in X are linearly dependent or nearly so.")


def vif(z_scores):
    check_linear_dependency(z_scores.values)
    vif_values = calculate_vif(z_scores)
    return vif_values


def calculate_vif(df):
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
