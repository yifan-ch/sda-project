import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_model
from data_model import df_mean, df_z_scores
from pathlib import Path
from env import PATHS
from multiple_regression_model import perform_regression_statsmodels, perform_regression_sklearn
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
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
    # randomly split data into two parts based on the fraction.
    def split(df, frac=0.5):
        p1 = df.sample(frac=frac)  # frac
        p2 = df.drop(p1.index)  # 1-frac

        return p2, p1

    def test():
        # split the data into two fractions
        df_0_training, df_0_test = split(
            data_model.status(df_z_scores, 0), frac_training
        )  # status 0
        df_1_training, df_1_test = split(
            data_model.status(df_z_scores, 1), frac_training
        )  # status 1

        # training data
        df_training = pd.concat([df_0_training, df_1_training], axis=0)

        y_training = df_training["status"].values
        X_training = df_training.drop(["status"], axis=1).values

        # test data
        df_test = pd.concat([df_0_test, df_1_test], axis=0)
        y_test = df_test["status"].values
        X_test = df_test.drop(["status"], axis=1).values

        model = perform_regression_sklearn(X_training, y_training)

        # predict
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # calc accuracy on a simple treshhold
        accuracy = np.sum([[y_pred >= tresh] == y_test]) / len(y_test)

        return mae, mse, rmse, r2, accuracy

    # perform multiple repetitions of the test and calc the mean
    mae, mse, rmse, r2, accuracy = np.mean(
        np.array([np.array(test()) for _ in range(repetitions)]), axis=0
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
