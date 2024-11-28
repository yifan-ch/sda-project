import pandas as pd
from pathlib import Path
from env import DATA_GENERATED_PATH


class Data:
    """
    Data framework for querying and manipulating parkinsons data
    """

    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.features = self.df.columns

        self.status_0 = self.df.loc[self.df["status"] == 0]
        self.status_1 = self.df.loc[self.df["status"] == 1]

    # def filter_feature(self, feature):
    #     return self.df[feature]

    def display(self, df):
        print(df.to_string())


data_raw = Data(Path(DATA_GENERATED_PATH) / "parkinsons_raw.csv")
data_mean = Data(Path(DATA_GENERATED_PATH) / "parkinsons_mean.csv")
