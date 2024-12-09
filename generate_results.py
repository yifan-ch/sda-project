import numpy as np
import matplotlib.pyplot as plt
import data_model
from data_model import df_z_scores
from pathlib import Path
from env import PATHS
from multiple_regression_model import perform_regression_statsmodels, test_regression_sklearn
from logistic_regression import run_logistic_regression
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
    df_z_scores, frac_training=0.5, threshold=0.5, repetitions=100, write=False
):
    # perform multiple repetitions of the test and calc the mean
    mae, mse, rmse, r2, accuracy = np.mean(
        np.array(
            [
                np.array(test_regression_sklearn(df_z_scores, frac_training, threshold))
                for _ in range(repetitions)
            ]
        ),
        axis=0,
    )

    if write:

        # Write result to file
        with open(
            PATHS["results"]["multiple-regression"] / "mulitple-regression-accuracy.txt", "w"
        ) as f:
            f.write(f"Mean absolute error: {mae}\n")
            f.write(f"Mean squared error: {mse}\n")
            f.write(f"Root mean squared error: {rmse}\n")
            f.write(f"R-squared (goodness-of-fit): {r2}\n")
            f.write(f"accuracy: {accuracy}\n")

    return accuracy


def plot_accuracy_over_frac(df, threshold, repetitions):
    fracs = np.linspace(0.1, 0.9, 20)
    accs = [
        perform_accuracy_multiple_regression(df, frac, threshold, repetitions) for frac in fracs
    ]

    plt.plot(fracs, accs, label="accuracy")
    plt.xlabel("fraction of training data")
    plt.ylabel("accuracy")
    plt.title(f"Accuracy as a function of training data fraction for threshold={threshold}")
    plt.legend()

    plt.savefig(PATHS["results"]["multiple-regression"] / "accuracy-over-fraction")
    plt.clf()


def plot_accuracy_over_thresh(df, frac, repetitions):
    threshs = np.linspace(0.1, 0.9, 20)
    accs = [
        perform_accuracy_multiple_regression(df, frac, threshold, repetitions)
        for threshold in threshs
    ]

    plt.plot(threshs, accs, label="accuracy")
    plt.xlabel("threshhold")
    plt.ylabel("accuracy")
    plt.title(f"Accuracy as a function of threshhold for training_data_fraction={frac}")
    plt.legend()

    plt.savefig(PATHS["results"]["multiple-regression"] / "accuracy-over-threshold")
    plt.clf()


if __name__ == "__main__":
    repetitions = 500
    threshhold = 0.5
    fraction_training = 0.5

    # if path doesnt exist, create all missing folders
    Path(PATHS["results"]["histogram"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["multiple-regression"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["vif"]).mkdir(parents=True, exist_ok=True)

    # plot_histogram(df_mean())
    # perform_vif(df_z_scores())
    # perform_multiple_regression(df_z_scores())
    perform_accuracy_multiple_regression(
        df_z_scores(), fraction_training, threshhold, repetitions, write=True
    )
    plot_accuracy_over_frac(df_z_scores(), threshold=threshhold, repetitions=repetitions)
    plot_accuracy_over_thresh(df_z_scores(), frac=fraction_training, repetitions=repetitions)
    run_logistic_regression()
