import pandas as pd
from pathlib import Path
from env import PATHS


class Data:
    """
    Data framework for querying and manipulating parkinsons data
    """

    # def __init__(self, path):
    #     self.df = pd.read_csv(path)

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


data_raw = Data.read("parkinsons_raw.csv")
data_raw = Data.read("parkinsons_mean.csv")
