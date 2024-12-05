import pandas as pd
from pathlib import Path
from env import PATHS


def read(filename, original=False):
    return pd.read_csv(PATHS["data"]["original" if original else "generated"] / filename)


def write(df, filename):
    # If path doesnt exist, create all missing folders
    Path(PATHS["data"]["generated"]).mkdir(parents=True, exist_ok=True)

    df.to_csv(Path(PATHS["data"]["generated"]) / filename, index=False)


def status(df, stat):
    return df.loc[df["status"] == stat]


def include(df, cols):
    return df[cols]


def exclude(df, cols):
    return df.drop(cols, axis=1)


def display(df):
    print(df.to_string())


df_raw = read("parkinsons_raw.csv")
df_mean = read("parkinsons_mean.csv")
df_mean_mean = read("parkinsons_mean_mean.csv")
df_z_scores = read("parkinsons_z_scores.csv")
