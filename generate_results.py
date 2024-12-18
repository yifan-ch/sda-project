"""
Performs tests for all models and writes the results to disk.
"""

import numpy as np
import matplotlib.pyplot as plt
import tools.data_tools as data_tools
from tools.data_tools import df_z_scores, df_subset_z_scores
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
from models.elastic_net_model import elastic_net_model
import argparse
import generate_data  # Import the generate_data module


# ---- HISTOGRAM ----


def plot_histogram(df, k=30):
    """
    Plot histograms for each feature in the dataframe, excluding the 'status' column.
    """
    # Drop the 'status' column from the feature list
    for feature in df.drop(["status"], axis=1).columns:
        bins = np.linspace(df[feature].min(), df[feature].max(), k)

        # Plot histograms for healthy and Parkinson's patients
        plt.hist(
            [data_tools.status(df, 0)[feature], data_tools.status(df, 1)[feature]],
            bins=bins,
            label=["Healthy (status=0)", "Parkinson's patient (status=1)"],
            density=True,
        )

        plt.title(feature)
        plt.legend()
        # Replace ':' with '-' in filenames
        plt.savefig(PATHS["results"]["histogram"] / feature.replace(":", "-"))
        plt.clf()


# ---- VIF ----


def perform_vif(df):
    """
    Perform Variance Inflation Factor (VIF) analysis and write the results to a file.
    """
    z = elastic_net_model(df)
    z["status"] = df["status"]
    z_scores = z
    vif_df = vif(z_scores)

    # Write VIF results to file
    with open(PATHS["results"]["vif"] / "vif.txt", "w") as f:
        f.writelines([f"{row['Predictor']}: {row['VIF']:.2f}\n" for _, row in vif_df.iterrows()])


# ---- MLR ----


def perform_mlr(z_scores):
    """
    Perform multiple linear regression and write the model summary to a file.
    """

    # Define dependent variable (Y) and independent variables (X)
    y = z_scores["status"].values
    X = z_scores.drop(["status"], axis=1).values

    model = perform_regression_statsmodels(X, y)

    # Write model summary to file
    with open(PATHS["results"]["multiple-regression"] / "multiple-regression.txt", "w") as f:
        print(model.summary(), file=f)


def perform_stats_mlr(
    df, fraction_training=0.5, threshold=0.5, repetitions=100, use_elasticnet=False
):
    """
    Perform statistical analysis for multiple linear regression and write the results to a file.
    """
    path = "multiple-regression"

    if use_elasticnet:
        path += "-elasticnet"

    accuracy, precision, recall, f1, TPR, FPR, FNR, TNR = stats_mlr(
        df, fraction_training, threshold, repetitions, use_elasticnet=use_elasticnet
    )

    # Write classification metrics to file
    with open(PATHS["results"][path] / f"{path}-classification.txt", "w") as f:
        f.write(
            f"stats for threshold={threshold}, fraction_training={fraction_training}, "
            f"repetitions={repetitions}\n\n"
        )

        f.write(f"accuracy (frac. correct):                                 {accuracy}\n")
        f.write(f"precision (frac. predicted positive actually positive):   {precision}\n")
        f.write(f"recall (frac. actual positives correctly identified):     {recall}\n")
        f.write(f"F1 (harmonic mean of precision and recall):               {f1}\n\n")

        f.write(f"TPR:                  {TPR}\n")
        f.write(f"FPR (type I error):   {FPR}\n")
        f.write(f"TNR:                  {TNR}\n")
        f.write(f"FNR (type II error):  {FNR}\n")


def plot_mlr_over_thres(df, fraction_training, repetitions, resolution=20, use_elasticnet=False):
    """
    Plot multiple linear regression metrics over a range of thresholds.
    """
    path = "multiple-regression"

    if use_elasticnet:
        path += "-elasticnet"

    thresholds = np.linspace(0.1, 0.9, resolution)
    accuracy, precision, recall, f1, TPR, FPR, FNR, TNR = zip(
        *[
            stats_mlr(df, fraction_training, threshold, repetitions, use_elasticnet=use_elasticnet)
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
            f"{', '.join([name.split(' ')[0] for name in names])} as a function of threshold\n"
            f"for training_data_fraction={fraction_training}, repetitions={repetitions}",
        )
        plt.legend()
        plt.savefig(
            PATHS["results"][path]
            / f"{'-'.join([name.split(' ')[0] for name in names])}-over-threshold",
            bbox_inches="tight",
        )
        plt.clf()

    # Plot different combinations of metrics
    plot(("accuracy", "TNR", "FNR"), (accuracy, TNR, FNR))
    plot(("accuracy", "TPR", "FPR"), (accuracy, TPR, FPR))
    plot(("accuracy", "FPR (type I error)", "FNR (type II error)"), (accuracy, FPR, FNR))
    plot(("accuracy", "precision", "recall", "f1"), (accuracy, precision, recall, f1))


# ---- LOGISTIC ----


def perform_logistic_regression(
    df,
    threshold=0.5,
    repetitions=100,
    epochs=1000,
    fraction_training=0.5,
    use_elasticnet=False,
):
    """
    Perform logistic regression and write the results to a file.
    """
    path = "logistic-regression"

    if use_elasticnet:
        path += "-elasticnet"

    metrics, losses = run_logistic_regression(
        df, threshold, repetitions, epochs, fraction_training, use_elasticnet=use_elasticnet
    )

    accuracy, precision, recall, f1, TPR, FPR, FNR, TNR = metrics

    # Write classification metrics to file
    with open(PATHS["results"][path] / f"{path}-classification.txt", "w") as f:
        f.write(
            f"stats for threshold={threshold}, fraction_training={fraction_training}, "
            f"repetitions={repetitions}\n\n"
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
    plt.title(f"Final Loss Across {repetitions} Runs")
    plt.savefig(PATHS["results"][path] / "loss-function.png", bbox_inches="tight")
    plt.clf()


def plot_logistic_regression_over_thres(
    df, fraction_training, repetitions, epochs, resolution=20, use_elasticnet=False
):
    """
    Plot logistic regression metrics over a range of thresholds.
    """
    path = "logistic-regression"

    if use_elasticnet:
        path += "-elasticnet"

    thresholds = np.linspace(0.1, 0.9, resolution)
    accuracy, precision, recall, f1, TPR, FPR, FNR, TNR = zip(
        *[
            run_logistic_regression(
                df,
                threshold,
                repetitions,
                epochs,
                fraction_training,
                use_elasticnet=use_elasticnet,
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
            f"{', '.join([name.split(' ')[0] for name in names])} as a function of threshold\n"
            f"for training_data_fraction={fraction_training}, repetitions={repetitions}",
        )
        plt.legend()
        plt.savefig(
            PATHS["results"][path]
            / f"{'-'.join([name.split(' ')[0] for name in names])}-over-threshold",
            bbox_inches="tight",
        )
        plt.clf()

    # Plot different combinations of metrics
    plot(("accuracy", "TNR", "FNR"), (accuracy, TNR, FNR))
    plot(("accuracy", "TPR", "FPR"), (accuracy, TPR, FPR))
    plot(("accuracy", "FPR (type I error)", "FNR (type II error)"), (accuracy, FPR, FNR))
    plot(("accuracy", "precision", "recall", "f1"), (accuracy, precision, recall, f1))


def plot_logistic_regression_accuracy_per_epoch(
    df, threshold=0.25, repetitions=300, fraction_training=0.5
):
    """
    Plot logistic regression accuracy and loss over a range of epochs.
    """
    iterations = np.arange(100, 4001, 100, dtype=int)
    accuracies = []
    final_losses = []

    for epochs in iterations:
        metrics, losses = run_logistic_regression(
            df,
            threshold,
            repetitions,
            epochs,
            fraction_training,
        )

        accuracies.append(metrics[0])
        final_losses.append(losses[-1])

    _, ax1 = plt.subplots()

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

    # Save the plot to a file
    plt.savefig(
        PATHS["results"]["logistic-regression"] / "accuracy_per_epoch.png", bbox_inches="tight"
    )
    plt.clf()


def plot_regressions_combined(df, repetitions, fraction_training, epochs, resolution=20):
    """
    Plot combined regression metrics for multiple linear regression and logistic regression.
    """
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
        plt.figtext(
            0, 0, f"for training_data_fraction={fraction_training}, repetitions={repetitions}"
        )
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Save the plot to a file
        plt.savefig(
            PATHS["results"]["regressions-combined"] / f"{title}-over-threshold",
            bbox_inches="tight",
        )
        plt.clf()

    # Collect data for each regression type
    data_mlr = tuple(
        zip(
            *[stats_mlr(df, fraction_training, threshold, repetitions) for threshold in thresholds]
        )
    )

    data_mlr_elasticnet = tuple(
        zip(
            *[
                stats_mlr(df, fraction_training, threshold, repetitions, use_elasticnet=True)
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
                    fraction_training,
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
                    fraction_training,
                    use_elasticnet=True,
                )[0]
                for threshold in thresholds
            ]
        )
    )

    # Plot each metric for all regression types
    for i, name in enumerate(
        ("Accuracy", "Precision", "Recall", "F1", "TPR", "FPR", "FNR", "TNR")
    ):
        data = data_mlr[i], data_mlr_elasticnet[i], data_logistic[i], data_logistic_elasticnet[i]
        plot(data, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate results for various models.")
    parser.add_argument("--repetitions", type=int, default=100, help="Number of repetitions")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold value")
    parser.add_argument(
        "--fraction_training", type=float, default=0.6, help="Fraction of training data"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--run_hist", action="store_true", help="Enable histogram generation")
    parser.add_argument("--run_vif", action="store_true", help="Enable VIF test")
    parser.add_argument("--run_mlr", action="store_true", help="Enable multiple linear regression")
    parser.add_argument("--run_logistic", action="store_true", help="Enable logistic regression")
    parser.add_argument(
        "--run_mlr_elasticnet", action="store_true", help="Enable MLR with Elastic Net"
    )
    parser.add_argument(
        "--run_logistic_elasticnet",
        action="store_true",
        help="Enable logistic regression with Elastic Net",
    )
    parser.add_argument(
        "--run_combined", action="store_true", help="Enable combined regression plots"
    )
    parser.add_argument(
        "--no_generate_data",
        action="store_true",
        help="Do not generate data before running models",
    )

    args = parser.parse_args()

    repetitions = args.repetitions
    threshold = args.threshold
    fraction_training = args.fraction_training
    epochs = args.epochs
    generate_data_flag = not args.no_generate_data  # Set generate_data_flag based on the argument

    # Determine which tests to run
    run_hist = args.run_hist
    run_vif = args.run_vif
    run_mlr = args.run_mlr
    run_logistic = args.run_logistic
    run_combined = args.run_combined
    run_mlr_elasticnet = args.run_mlr_elasticnet
    run_logistic_elasticnet = args.run_logistic_elasticnet

    if generate_data_flag:
        print("Generating data...")
        generate_data.generate_data()

    # If no enable flags are given, run all tests
    if not any([run_hist, run_vif, run_mlr, run_logistic, run_combined]):
        run_hist = run_vif = run_mlr = run_logistic = run_combined = True

    # if path doesnt exist, create all missing folders
    Path(PATHS["results"]["histogram"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["multiple-regression"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["multiple-regression-elasticnet"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["logistic-regression"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["logistic-regression-elasticnet"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["regressions-combined"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["results"]["vif"]).mkdir(parents=True, exist_ok=True)

    print("-------------------------------------------")
    print("User parameters")
    print("-------------------------------------------")
    print()
    print(f"Repetitions: {repetitions}")
    print(f"Threshold: {threshold}")
    print(f"Fraction training: {fraction_training}")
    print(f"Epochs: {epochs}")
    print(f"Enable histogram: {run_hist}")
    print(f"Enable VIF: {run_vif}")
    print(f"Enable MLR: {run_mlr}")
    print(f"Enable Logistic Regression: {run_logistic}")
    print(f"Enable MLR with Elastic Net: {run_mlr_elasticnet}")
    print(f"Enable Logistic Regression with Elastic Net: {run_logistic_elasticnet}")
    print(f"Enable Combined Plots: {run_combined}")
    print()
    print("-------------------------------------------")
    print("Generating results, this may take a while...")
    print("-------------------------------------------")
    print()

    # -- Histograms

    if run_hist:
        print("Plotting histograms...")

        plot_histogram(df_z_scores())

    # -- Vif

    if run_vif:
        print("Performing VIF test...")

        perform_vif(df_z_scores())

    # -- Multiple regression

    if run_mlr:
        print("Performing MLR...")

        perform_mlr(df_subset_z_scores())
        perform_stats_mlr(df_z_scores(), fraction_training, threshold, repetitions)
        plot_mlr_over_thres(df_z_scores(), fraction_training, repetitions)

    if run_mlr_elasticnet:
        print("Performing MLR with Elastic Net...")

        perform_stats_mlr(
            df_z_scores(), fraction_training, threshold, repetitions, use_elasticnet=True
        )
        plot_mlr_over_thres(
            df_z_scores(),
            fraction_training,
            repetitions,
            use_elasticnet=True,
        )

    # -- Logistic regression

    if run_logistic:
        print("Performing Logistic regression...")

        perform_logistic_regression(
            df_z_scores(),
            threshold,
            repetitions,
            epochs,
            fraction_training,
        )
        plot_logistic_regression_over_thres(df_z_scores(), fraction_training, repetitions, epochs)
        plot_logistic_regression_accuracy_per_epoch(
            df_z_scores(),
            threshold,
            repetitions,
            fraction_training,
        )

    if run_logistic_elasticnet:
        print("Performing Logistic regression with Elastic Net...")

        perform_logistic_regression(
            df_z_scores(),
            threshold,
            repetitions,
            epochs,
            fraction_training,
            use_elasticnet=True,
        )

        plot_logistic_regression_over_thres(
            df_z_scores(), fraction_training, repetitions, epochs, use_elasticnet=True
        )

    # -- All regressions combined in one plot

    if run_combined:
        print("Plotting combined stats for all regressions...")
        plot_regressions_combined(df_z_scores(), repetitions, fraction_training, epochs)

    print("All done! See results in the results folder.")
