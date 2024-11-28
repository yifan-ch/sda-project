import numpy as np
import matplotlib.pyplot as plt


import data_model

data = data_model.data_mean


def plot(data, feature, k=30):
    bins = np.linspace(data.df[feature].min(), data.df[feature].max(), k)

    plt.hist(
        [data.filter_status(0), data.filter_status(1)],
        bins=bins,
        label=["Healthy (status=0)", "Parkinson's patient (status=1)"],
    )

    plt.title(feature)
    plt.legend()
    plt.savefig(f"results/visualization/{feature}")
    plt.clf()


for f in data.features:
    plot(data.filter_status(0), data.filter_status(1), f)
