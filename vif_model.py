import numpy as np


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


def calculate_vif(X):
    n_predictors = X.shape[1]
    vif = []
    for i in range(n_predictors):
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)
        try:
            beta = np.linalg.lstsq(X_other, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            print(f"Error calculating VIF for predictor {i}. Skipping...")
            vif.append(np.inf)
            continue
        y_pred = X_other @ beta
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        vif.append(1 / (1 - r_squared) if 1 - r_squared > 0 else np.inf)
    return vif


def standardize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


def check_linear_dependency(X):
    singular_values = np.linalg.svd(X, compute_uv=False)
    print("Singular values:", singular_values)
    if np.min(singular_values) < 1e-10:
        print("Warning: Columns in X are linearly dependent or nearly so.")


def vif(z_scores):
    # Get predictors
    X = z_scores.values

    # Check for linear dependency
    check_linear_dependency(X)

    # Compute the correlation matrix
    correlation_matrix = calculate_correlation_matrix(X)
    # print("Correlation Matrix:")
    # print(correlation_matrix)

    # Compute VIFs
    vif_values = calculate_vif(X)
    # print("VIFs:")
    # for i, var in enumerate(z_scores.columns):
    # print(f"{var}: {vif_values[i]:.2f}")

    # return [f"{var}: {vif_values[i]:.2f}\n" for i, var in enumerate()]

    return z_scores.columns, vif_values
