import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Data:
    def __init__(self, path):
        self.df = pd.read_csv(path).drop("name", axis=1)


data = Data("data/parkinsons_data.csv")
# data = Data("data/test.csv")

df = data.df
features = df.columns


def plot(df, feature):
    k = 30
    bins = np.linspace(df[feature].min(), df[feature].max(), k)

    d1 = df.loc[df["status"] == 0][feature]
    d2 = df.loc[df["status"] == 1][feature]

    # plt.hist(d1, bins=bins, label="Healthy (status=0)", alpha=0.5)
    # plt.hist(d2, bins=bins, label="Parkinson's (status=1)", alpha=0.5)

    plt.hist([d1, d2], bins=bins, label=["Healthy (status=0)", "Parkinson's (status=1)"])

    plt.title(feature)
    plt.legend()
    plt.savefig(f"results/visualization/{feature}")
    plt.clf()
    # plt.show()


for f in features:
    plot(df, f)
