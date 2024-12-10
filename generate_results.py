import numpy as np
import matplotlib.pyplot as plt
import data_model
from data_model import df_mean, df_z_scores
from pathlib import Path
from env import PATHS
from multiple_regression_model import perform_regression_statsmodels, test_regression_sklearn

from vif_model import vif


def plot_histogram(df, k=30):
    # drop the status col in the feature list
    for feature in df.drop(["status"], axis=1).columns:
        bins = np.linspace(df[feature].min(), df[feature].max(), k)

        plt.hist(
            [data_model.status(df, 0)[feature], data_model.status(df, 1)[feature]],
            bins=bins,
            label=["Healthy (status=0)", "Parkinson's patient (status=1)"],
            density=True,
        )

        plt.title(feature)
        plt.legend()
        # : is not allowed in filenames
        plt.savefig(PATHS["results"]["histogram"] / feature.replace(":", "-"))
        plt.clf()


def perform_multiple_regression(z_scores):
    """Multiple linear regression"""
    # Define dependent variable (Y) and independent variables (X)
    y = z_scores["status"].values
    X = z_scores.drop(["status"], axis=1).values

    model = perform_regression_statsmodels(X, y)

    # Write result to file
    with open(PATHS["results"]["multiple-regression"] / "multiple-regression.txt", "w") as f:
        print(model.summary(), file=f)


def perform_vif(df_z_scores):
    columns, vif_values = vif(df_z_scores)
    # Write result to file
    with open(PATHS["results"]["vif"] / "vif.txt", "w") as f:
        f.writelines([f"{c}: {v:.2f}\n" for c, v in zip(columns, vif_values)])


def perform_accuracy_multiple_regression(
    df_z_scores, frac_training=0.5, tresh=0.5, repetitions=100
):
    # perform multiple repetitions of the test and calc the mean
    mae, mse, rmse, r2, accuracy = np.mean(
        np.array(
            [
                np.array(test_regression_sklearn(df_z_scores, frac_training, tresh))
                for _ in range(repetitions)
            ]
        ),
        axis=0,
    )

    # Write result to file
    with open(
        PATHS["results"]["multiple-regression"] / "mulitple-regression-accuracy.txt", "w"
    ) as f:
        f.write(f"Mean absolute error: {mae}\n")
        f.write(f"Mean squared error: {mse}\n")
        f.write(f"Root mean squared error: {rmse}\n")
        f.write(f"R-squared (goodness-of-fit): {r2}\n")
        f.write(f"accuracy: {accuracy}\n")


if __name__ == "__main__":
    # if path doesnt exist, create all missing folders
    Path(PATHS["results"]["histogram"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["multiple-regression"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["vif"]).mkdir(parents=True, exist_ok=True)

    plot_histogram(df_mean())
    perform_vif(df_z_scores())
    perform_multiple_regression(df_z_scores())
    perform_accuracy_multiple_regression(df_z_scores(), 0.5, 0.5, 1000)
