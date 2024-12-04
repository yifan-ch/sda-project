import numpy as np
import pandas as pd
from pathlib import Path
from env import DATA_PATH, DATA_GENERATED_PATH

import scipy

def write(df, filename):
    df.to_csv(Path(DATA_GENERATED_PATH) / filename, index=False)

# Generate custom csv data from the original
def generate_data():
    # If path doesnt exist, create all missing folders
    Path(DATA_GENERATED_PATH).mkdir(parents=True, exist_ok=True)

    # Drop name column, as they don't contain important info
    df = pd.read_csv(Path(DATA_PATH) / "parkinsons_original.csv").drop("name", axis=1)

    # Create subject id and experiment number columns for easier indexing.
    df.insert(0, "experiment", np.tile(np.arange(0, 6), len(df) // 6))
    df.insert(0, "id", np.repeat(np.arange(0, len(df) // 6), 6))
    # Add variables vocal range and relative pitch
    df.insert(5, "vocal_range", df["MDVP:Fhi(Hz)"] - df["MDVP:Flo(Hz)"])
    df.insert(6, "relative_avg_pitch", ((df["MDVP:Fo(Hz)"] - df["MDVP:Flo(Hz)"]) / (df["MDVP:Fhi(Hz)"] - df["MDVP:Flo(Hz)"])))
    write(df, "parkinsons_raw.csv")

    # Take the mean of the experiments
    grouped_df = df.groupby("id", as_index=False).mean()
    write(grouped_df, "parkinsons_mean.csv")

    # Calculate the mean of the means and standard deviation
    grouped_df_mean = grouped_df.mean()
    grouped_df_mean_df = grouped_df_mean.to_frame(name="mean").reset_index()
    grouped_df_mean_df["standard_deviation"] = grouped_df.std().values
    write(grouped_df_mean_df, "parkinsons_mean_mean.csv")

    # Calculate the overall mean and standard deviation of the means
    mean_of_means = grouped_df.mean()
    std_of_means = grouped_df.std()

    # Calculate z-scores for each person
    z_scores = (grouped_df - mean_of_means) / std_of_means

    # Save the z-scores to a new CSV file
    write(z_scores.drop(["id", "experiment"], axis=1), "parkinsons_z_scores.csv")
    return z_scores

def calculate_correlation_matrix(X):
    n, m = X.shape
    correlation_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            X_i, X_j = X[:, i], X[:, j]
            mean_i, mean_j = np.mean(X_i), np.mean(X_j)
            covariance = np.sum((X_i - mean_i) * (X_j - mean_j)) / n
            std_i = np.sqrt(np.sum((X_i - mean_i)**2) / n)
            std_j = np.sqrt(np.sum((X_j - mean_j)**2) / n)
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
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred)**2)
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


if __name__ == "__main__":
    z_scores = generate_data()

    # Get predictors
    X = z_scores.drop(["id", "experiment"], axis=1).values

    # Check for linear dependency
    check_linear_dependency(X)

    # Compute the correlation matrix
    correlation_matrix = calculate_correlation_matrix(X)
    print("Correlation Matrix:")
    print(correlation_matrix)

    # Compute VIFs
    vif_values = calculate_vif(X)
    print("VIFs:")
    for i, var in enumerate(z_scores.drop(["id", "experiment"], axis=1).columns):
        print(f"{var}: {vif_values[i]:.2f}")
