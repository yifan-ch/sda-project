import pandas as pd
from pathlib import Path
from env import PATHS


def read(filename, original=False, sep=","):
    return pd.read_csv(PATHS["data"]["original" if original else "generated"] / filename, sep=sep)


def write(df, filename):
    # If path doesnt exist, create all missing folders
    Path(PATHS["data"]["generated"]).mkdir(parents=True, exist_ok=True)

    df.to_csv(Path(PATHS["data"]["generated"]) / filename, index=False)


def status(df, stat):
    return df.loc[df["status"] == stat]


def display(df):
    print(df.to_string())


def df_z_scores():
    df = read("parkinsons_z_scores.csv")

    # df2 = df.drop(["status"], axis=1)
    # # to fix 0 divide
    # df2 += 1e-9

    # df2["status"] = df["status"]

    # return df2

    return df
