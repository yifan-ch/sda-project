"""
Performs tests for all models and writes the results to disk.
"""

import numpy as np
import matplotlib.pyplot as plt
import tools.data_tools as data_tools
from tools.data_tools import df_z_scores
from pathlib import Path
from env import PATHS
from models.multiple_regression_model import (
    perform_regression_statsmodels,
    stats_mlr,
)
from models.logistic_regression_model import (
    run_logistic_regression,
)
from models.vif_model import vif

# ---- HISTOGRAM ----


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


# ---- VIF ----


def perform_vif(df):
    columns, vif_values = vif(df)
    # Write result to file
    with open(PATHS["results"]["vif"] / "vif.txt", "w") as f:
        f.writelines([f"{c}: {v:.2f}\n" for c, v in zip(columns, vif_values)])


# ---- MLR ----


def perform_mlr(z_scores):
    """Multiple linear regression"""

    # Define dependent variable (Y) and independent variables (X)
    y = z_scores["status"].values
    X = z_scores.drop(["status"], axis=1).values

    model = perform_regression_statsmodels(X, y)

    # Write result to file
    with open(PATHS["results"]["multiple-regression"] / "multiple-regression.txt", "w") as f:
        print(model.summary(), file=f)


def perform_stats_mlr(df, frac_training=0.5, threshold=0.5, repetitions=100, use_elasticnet=False):
    path = "multiple-regression"

    if use_elasticnet:
        path += "-elasticnet"

    accuracy, precision, recall, f1, TPR, FPR, FNR, TNR = stats_mlr(
        df, frac_training, threshold, repetitions, use_elasticnet=use_elasticnet
    )

    with open(PATHS["results"][path] / f"{path}-classification.txt", "w") as f:
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


def plot_mlr_over_thres(df, frac, repetitions, resolution=20, use_elasticnet=False):
    path = "multiple-regression"

    if use_elasticnet:
        path += "-elasticnet"

    thresholds = np.linspace(0.1, 0.9, resolution)
    accuracy, precision, recall, f1, TPR, FPR, FNR, TNR = zip(
        *[
            stats_mlr(df, frac, threshold, repetitions, use_elasticnet=use_elasticnet)
            for threshold in thresholds
        ]
    )

    def plot(names, stats):
        colors = ("blue", "green", "red", "orange")[: len(names)]
        for name, stat, color in zip(names, stats, colors):
            plt.plot(thresholds, stat, label=name, color=color)
            plt.xlabel("threshold")
            plt.ylabel("value")

        plt.title(
            "Metrics for Multiple linear regression"
            + (" with Elastic Net" if use_elasticnet else "")
        )
        plt.figtext(
            0,
            -0.05,
            f"{', '.join([name.split(' ')[0] for name in names])} as a function of threshold\nfor training_data_fraction={frac}, repetitions={repetitions}",
        )
        plt.legend()
        plt.savefig(
            PATHS["results"][path]
            / f"{'-'.join([name.split(' ')[0] for name in names])}-over-threshold",
            bbox_inches="tight",
        )
        plt.clf()

    plot(("accuracy", "TNR", "FNR"), (accuracy, TNR, FNR))
    plot(("accuracy", "TPR", "FPR"), (accuracy, TPR, FPR))
    plot(("accuracy", "FPR (type I error)", "FNR (type II error)"), (accuracy, FPR, FNR))

    plot(("accuracy", "precision", "recall", "f1"), (accuracy, precision, recall, f1))


# ---- LOGISTIC ----


def perform_logistic_regression(
    df, threshold=0.5, reps=100, epochs=1000, frac_training=0.5, use_elasticnet=False
):
    path = "logistic-regression"

    if use_elasticnet:
        path += "-elasticnet"

    metrics, losses = run_logistic_regression(
        df, threshold, reps, epochs, frac_training, use_elasticnet=use_elasticnet
    )

    accuracy, precision, recall, f1, TPR, FPR, FNR, TNR = metrics

    with open(PATHS["results"][path] / f"{path}-classification.txt", "w") as f:
        f.write(
            f"stats for threshold={threshold}, fraction_training={frac_training}, repetitions={reps}\n\n"
        )

        f.write(f"accuracy (frac. correct):                                 {accuracy}\n")
        f.write(f"precision (frac. predicted positive actually positive):   {precision}\n")
        f.write(f"recall (frac. actual positives correctly identified):     {recall}\n")
        f.write(f"F1 (harmonic mean of precision and recall):               {f1}\n\n")

        f.write(f"TPR:                  {TPR}\n")
        f.write(f"FPR (type I error):   {FPR}\n")
        f.write(f"TNR:                  {TNR}\n")
        f.write(f"FNR (type II error):  {FNR}\n")

    # Plot the final loss curve for all runs
    plt.plot(losses)
    plt.xlabel("Run Index")
    plt.ylabel("Final Loss")
    plt.title(f"Final Loss Across {reps} Runs")
    plt.savefig(PATHS["results"][path] / "loss-function.png", bbox_inches="tight")
    plt.clf()


def plot_logistic_regression_over_thres(
    df, frac, repetitions, epochs, resolution=20, use_elasticnet=False
):
    path = "logistic-regression"

    if use_elasticnet:
        path += "-elasticnet"

    thresholds = np.linspace(0.1, 0.9, resolution)
    accuracy, precision, recall, f1, TPR, FPR, FNR, TNR = zip(
        *[
            run_logistic_regression(
                df, threshold, repetitions, epochs, frac, use_elasticnet=use_elasticnet
            )[0]
            for threshold in thresholds
        ]
    )

    def plot(names, stats):
        colors = ("blue", "green", "red", "orange")[: len(names)]
        for name, stat, color in zip(names, stats, colors):
            plt.plot(thresholds, stat, label=name, color=color)
            plt.xlabel("threshold")
            plt.ylabel("value")

        plt.title(
            "Metrics for Logistic regression" + (" with Elastic Net" if use_elasticnet else "")
        )
        plt.figtext(
            0,
            -0.05,
            f"{', '.join([name.split(' ')[0] for name in names])} as a function of threshold\nfor training_data_fraction={frac}, repetitions={repetitions}",
        )
        plt.legend()
        plt.savefig(
            PATHS["results"][path]
            / f"{'-'.join([name.split(' ')[0] for name in names])}-over-threshold",
            bbox_inches="tight",
        )
        plt.clf()

    plot(("accuracy", "TNR", "FNR"), (accuracy, TNR, FNR))
    plot(("accuracy", "TPR", "FPR"), (accuracy, TPR, FPR))
    plot(("accuracy", "FPR (type I error)", "FNR (type II error)"), (accuracy, FPR, FNR))

    plot(("accuracy", "precision", "recall", "f1"), (accuracy, precision, recall, f1))


def plot_logistic_regression_accuracy_per_epoch(
    df, threshold=0.25, num_reps=300, frac_training=0.5
):
    iterations = np.arange(100, 4001, 100, dtype=int)
    accuracies = []
    final_losses = []

    for num_epochs in iterations:
        accuracy, losses = run_logistic_regression(
            df,
            threshold=threshold,
            num_reps=num_reps,
            num_epochs=num_epochs,
            frac_training=frac_training,
        )
        accuracies.append(accuracy)
        final_losses.append(losses[-1])

    fig, ax1 = plt.subplots()

    # Plot accuracies on the primary y-axis
    ax1.plot(iterations, accuracies, label="Accuracy", color="blue", marker="o")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_title("Accuracy and Loss vs. Epochs")

    # Plot losses on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(iterations, final_losses, label="Loss", color="red", marker="x")
    ax2.set_ylabel("Loss", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Add legends
    plt.savefig(
        PATHS["results"]["logistic-regression"] / "accuracy_per_epoch.png", bbox_inches="tight"
    )
    plt.show()


def plot_regressions_combined(df, repetitions, frac_training, epochs, resolution=20):
    thresholds = np.linspace(0.1, 0.9, resolution)

    def plot(data, title):
        names = (
            "Multiple regression",
            "Multiple regression elasticnet",
            "Logistic regresssion",
            "Logistic regression elastic net",
        )

        colors = ("blue", "green", "red", "orange")[: len(names)]
        for name, d, color in zip(names, data, colors):
            plt.plot(thresholds, d, label=name, color=color)
            plt.xlabel("threshold")
            plt.ylabel("value")

        plt.title(f"{title} as a function of threshold")
        plt.figtext(0, 0, f"for training_data_fraction={frac_training}, repetitions={repetitions}")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        plt.savefig(
            PATHS["results"]["regressions-combined"] / f"{title}-over-threshold",
            bbox_inches="tight",
        )
        plt.clf()

    data_mlr = tuple(
        zip(*[stats_mlr(df, frac_training, threshold, repetitions) for threshold in thresholds])
    )

    data_mlr_elasticnet = tuple(
        zip(
            *[
                stats_mlr(df, frac_training, threshold, repetitions, use_elasticnet=True)
                for threshold in thresholds
            ]
        )
    )

    data_logistic = tuple(
        zip(
            *[
                run_logistic_regression(
                    df,
                    threshold,
                    repetitions,
                    epochs,
                    frac_training=frac_training,
                )[0]
                for threshold in thresholds
            ]
        )
    )

    data_logistic_elasticnet = tuple(
        zip(
            *[
                run_logistic_regression(
                    df,
                    threshold,
                    repetitions,
                    epochs,
                    frac_training=frac_training,
                    use_elasticnet=True,
                )[0]
                for threshold in thresholds
            ]
        )
    )

    for i, name in enumerate(
        ("Accuracy", "Precision", "Recall", "F1", "TPR", "FPR", "FNR", "TNR")
    ):
        data = data_mlr[i], data_mlr_elasticnet[i], data_logistic[i], data_logistic_elasticnet[i]
        plot(data, name)


if __name__ == "__main__":
    # Parameters that can be changed
    repetitions = 100
    threshold = 0.5
    fraction_training = 0.6
    epochs = 100

    # Enable or disable certain tests
    enable_hist = False
    enable_vif = False
    enable_mlr = True
    enable_logistic = True
    enable_combined = True

    # if path doesnt exist, create all missing folders
    Path(PATHS["results"]["histogram"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["multiple-regression"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["multiple-regression-elasticnet"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["logistic-regression"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["logistic-regression-elasticnet"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["regressions-combined"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["vif"]).mkdir(parents=True, exist_ok=True)

    # -- Histograms

    if enable_hist:
        print("Plotting histograms...")
        plot_histogram(df_z_scores())

    # -- Vif

    if enable_vif:
        print("Performing VIF test...")
        perform_vif(df_z_scores())

    # -- Multiple regression

    if enable_mlr:
        print("Performing MLR...")
        perform_mlr(df_z_scores())

        perform_stats_mlr(df_z_scores(), fraction_training, threshold, repetitions)
        # perform_stats_mlr(
        #     df_z_scores(), fraction_training, threshold, repetitions, use_elasticnet=True
        # )

        plot_mlr_over_thres(df_z_scores(), frac=fraction_training, repetitions=repetitions)
        # plot_mlr_over_thres(
        #     df_z_scores(), frac=fraction_training, repetitions=repetitions, use_elasticnet=True
        # )

    # -- Logistic regression

    if enable_logistic:
        print("Performing Logistic regression...")
        perform_logistic_regression(
            df_z_scores(),
            threshold,
            repetitions,
            epochs,
            fraction_training,
        )
        perform_logistic_regression(
            df_z_scores(),
            threshold,
            repetitions,
            epochs,
            fraction_training,
            use_elasticnet=True,
        )

        plot_logistic_regression_over_thres(df_z_scores(), fraction_training, repetitions, epochs)

        plot_logistic_regression_over_thres(
            df_z_scores(), fraction_training, repetitions, epochs, use_elasticnet=True
        )

        # plot_logistic_regression_accuracy_per_epoch(
        #     df_z_scores(),
        #     threshold=threshold,
        #     num_reps=repetitions,
        #     frac_training=fraction_training,
        # )

    # -- All regressions combined in one plot

    if enable_combined:
        print("Plotting combined stats for all regressions...")
        plot_regressions_combined(df_z_scores(), repetitions, fraction_training, epochs)
