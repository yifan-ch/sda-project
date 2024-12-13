"""
Performs tests for all models and writes the results to disk.
"""

import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import cm
import tools.data_tools as data_tools
from tools.data_tools import df_z_scores
from pathlib import Path
from env import PATHS
from models.multiple_regression_model import perform_regression_statsmodels, stats_mlr
from models.logistic_regression_model import run_logistic_regression
from models.vif_model import vif

# from itertools import product


def plot_histogram(df, k=30):
    # drop the status col in the feature list
    for feature in df.drop(["status"], axis=1).columns:
        bins = np.linspace(df[feature].min(), df[feature].max(), k)

        plt.hist(
            [data_tools.status(df, 0)[feature], data_tools.status(df, 1)[feature]],
            bins=bins,
            label=["Healthy (status=0)", "Parkinson's patient (status=1)"],
            density=True,
        )

        plt.title(feature)
        plt.legend()
        # : is not allowed in filenames
        plt.savefig(PATHS["results"]["histogram"] / feature.replace(":", "-"))
        plt.clf()


def perform_mlr(z_scores):
    """Multiple linear regression"""
    # Define dependent variable (Y) and independent variables (X)
    y = z_scores["status"].values
    X = z_scores.drop(["status"], axis=1).values

    model = perform_regression_statsmodels(X, y)

    # Write result to file
    with open(PATHS["results"]["multiple-regression"] / "multiple-regression.txt", "w") as f:
        print(model.summary(), file=f)


def perform_vif(df):
    columns, vif_values = vif(df)
    # Write result to file
    with open(PATHS["results"]["vif"] / "vif.txt", "w") as f:
        f.writelines([f"{c}: {v:.2f}\n" for c, v in zip(columns, vif_values)])


def perform_stats_mlr(df, frac_training=0.5, threshold=0.5, repetitions=100):
    accuracy, precision, recall, f1, TPR, FPR, FNR, TNR = stats_mlr(
        df, frac_training, threshold, repetitions
    )

    with open(
        PATHS["results"]["multiple-regression"] / "mulitple-regression-classification.txt", "w"
    ) as f:
        # f.write(f"Mean absolute error: {mae}\n")
        # f.write(f"Mean squared error: {mse}\n")
        # f.write(f"Root mean squared error: {rmse}\n")
        # f.write(f"R-squared (goodness-of-fit): {r2}\n")
        # f.write(f"accuracy: {accuracy}\n")

        f.write(
            f"stats for threshold={threshold}, fraction_training={frac_training}, repetitions={repetitions}\n\n"
        )

        f.write(f"accuracy (frac. correct):                                 {accuracy}\n")
        f.write(f"precision (frac. predicted positive actually positive):   {precision}\n")
        f.write(f"recall (frac. actual positives correctly identified):     {recall}\n")
        f.write(f"F1 (harmonic mean of precision and recall):               {f1}\n\n")

        f.write(f"TPR:                  {TPR}\n")
        f.write(f"FPR (type I error):   {FPR}\n")
        f.write(f"TNR:                  {TNR}\n")
        f.write(f"FNR (type II error):  {FNR}\n")


# def plot_accuracy_over_frac(df, threshold, repetitions, resolution=20):
#     fracs = np.linspace(0.1, 0.9, resolution)
#     accs = [perform_stats_mlr(df, frac, threshold, repetitions) for frac in fracs]

#     plt.plot(fracs, accs, label="accuracy")
#     plt.xlabel("fraction of training data")
#     plt.ylabel("accuracy")
#     plt.title(f"Accuracy as a function of training data fraction for threshold={threshold}")
#     plt.legend()

#     plt.savefig(PATHS["results"]["multiple-regression"] / "accuracy-over-fraction")
#     plt.clf()


def plot_mlr_over_thres(df, frac, repetitions, resolution=20):
    thresholds = np.linspace(0.1, 0.9, resolution)
    accuracy, precision, recall, f1, TPR, FPR, FNR, TNR = zip(
        *[stats_mlr(df, frac, threshold, repetitions) for threshold in thresholds]
    )

    def plot(names, stats):
        colors = ("blue", "green", "red", "orange")[: len(names)]
        for name, stat, color in zip(names, stats, colors):
            plt.plot(thresholds, stat, label=name, color=color)
            plt.xlabel("threshold")
            plt.ylabel("value")

        plt.title(
            f"{', '.join([name.split(' ')[0] for name in names])} as a function of threshold"
        )
        plt.figtext(0, 0, f"for training_data_fraction={frac}, repetitions={repetitions}")
        plt.legend()
        plt.savefig(
            PATHS["results"]["multiple-regression"]
            / f"{'-'.join([name.split(' ')[0] for name in names])}-over-threshold"
        )
        plt.clf()

    plot(("accuracy", "TNR", "FNR"), (accuracy, TNR, FNR))
    plot(("accuracy", "TPR", "FPR"), (accuracy, TPR, FPR))
    plot(("accuracy", "FPR (type I error)", "FNR (type II error)"), (accuracy, FPR, FNR))

    plot(("accuracy", "precision", "recall", "f1"), (accuracy, precision, recall, f1))


# def plot_mlr_over_frac_thres(df, repetitions, resolution=20):
#     thresholds = fracs = np.linspace(0.1, 0.9, resolution)  # Define thresholds and fractions
#     threshold_grid, frac_grid = np.meshgrid(thresholds, fracs)  # Create a meshgrid

#     # Compute accuracies for each pair (threshold, fraction)
#     accuracies = np.array(
#         [stats_mlr(df, f, t, repetitions) for t, f in product(thresholds, fracs)]
#     ).reshape(
#         len(fracs), len(thresholds)
#     )  # Reshape to match the grid dimensions

#     # Create 3D plot
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     surf = ax.plot_surface(
#         threshold_grid, frac_grid, accuracies, cmap=cm.coolwarm, linewidth=0, antialiased=False
#     )
#     # ax.plot_wireframe(threshold_grid, frac_grid, accuracies, cmap=cm.coolwarm, rstride=1, cstride=1)

#     ax.set_xlabel("threshold")
#     ax.set_ylabel("training data fraction")
#     ax.set_zlabel("accuracy")

#     ax.set_title("accuracy as a function of threshold and fraction of training data")

#     # Add color bar for reference
#     cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
#     cbar.set_label("accuracy")
#     plt.show()


if __name__ == "__main__":
    repetitions = 100
    threshold = 0.5
    fraction_training = 0.6

    # if path doesnt exist, create all missing folders
    Path(PATHS["results"]["histogram"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["multiple-regression"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["logistic-regression"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["elasticnet-regression"]).mkdir(parents=True, exist_ok=True)

    Path(PATHS["results"]["vif"]).mkdir(parents=True, exist_ok=True)

    # plot_histogram(df_mean())
    # perform_vif(df_z_scores())
    # perform_mlr(df_z_scores())
    perform_stats_mlr(df_z_scores(), fraction_training, threshold, repetitions)
    # plot_accuracy_over_frac(df_z_scores(), threshold=threshhold, repetitions=repetitions)

    plot_mlr_over_thres(df_z_scores(), frac=fraction_training, repetitions=repetitions)
    # plot_accuracy_over_frac_thres(df_z_scores(), repetitions=100)
    # run_logistic_regression(threshold=0.25, num_reps=10)
