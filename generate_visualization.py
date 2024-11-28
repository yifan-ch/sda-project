import numpy as np
import matplotlib.pyplot as plt
import data_model
from pathlib import Path
from env import RESULTS_VISUALIZATION_PATH

# if path doesnt exist, create all missing folders
Path(RESULTS_VISUALIZATION_PATH).mkdir(parents=True, exist_ok=True)


data = data_model.data_mean


def plot(data, feature, k=30):
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
    plt.savefig(Path(RESULTS_VISUALIZATION_PATH) / feature.replace(":", "-"))
    plt.clf()


if __name__ == "__main__":
    for feature in data.features:
        plot(data, feature)
