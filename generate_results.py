import numpy as np
import matplotlib.pyplot as plt
import data_model
from pathlib import Path
from env import RESULTS_FEATURES_PATH

# if path doesnt exist, create all missing folders
Path(RESULTS_FEATURES_PATH).mkdir(parents=True, exist_ok=True)


data = data_model.data_mean


def plot_features(data, feature, k=30):
    # skip meaningless stats
    if feature in ("id", "experiment", "status"):
        return

    bins = np.linspace(data.df[feature].min(), data.df[feature].max(), k)

    plt.hist(
        [data.status_0[feature], data.status_1[feature]],
        bins=bins,
        label=["Healthy (status=0)", "Parkinson's patient (status=1)"],
    )

    plt.title(feature)
    plt.legend()
    plt.savefig(Path(RESULTS_FEATURES_PATH) / feature.replace(":", "-"))
    plt.clf()


def plot_linear_regression(data):
    pass


if __name__ == "__main__":
    for feature in data.features:
        plot_features(data, feature)
