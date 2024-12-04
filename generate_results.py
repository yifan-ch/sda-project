import numpy as np
import matplotlib.pyplot as plt
from data_model import Data, data_raw
from pathlib import Path
from env import PATHS

# if path doesnt exist, create all missing folders
Path(PATHS["results"]["histogram"]).mkdir(parents=True, exist_ok=True)
Path(PATHS["results"]["linear-regression"]).mkdir(parents=True, exist_ok=True)
Path(PATHS["results"]["multiple-regression"]).mkdir(parents=True, exist_ok=True)

df = data_raw


def plot_features(df, k=20):
    df.drop(["id", "status"], axis=1)

    for feature in df.columns:
        bins = np.linspace(df[feature].min(), df[feature].max(), k)

        plt.hist(
            [Data.status(df, 0)[feature], Data.status(df, 1)[feature]],
            bins=bins,
            label=["Healthy (status=0)", "Parkinson's patient (status=1)"],
            density=True,
        )

        plt.title(feature)
        plt.legend()
        plt.savefig(PATHS["results"]["histogram"] / feature.replace(":", "-"))
        plt.clf()


def plot_linear_regression(df):
    pass


if __name__ == "__main__":
    plot_features(df)
