import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/parkinsons_data.csv")
features = df.columns[1:]  # .tolist()


def plot(df, feature):
    print(df[feature])


plot(df, features[0])
