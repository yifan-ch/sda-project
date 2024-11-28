import pandas as pd
from env import DATA_PATH


class Data:
    """
    Data framework for querying and manipulating parkinsons data
    """

    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.features = self.df.columns

    def filter_status(self, status):
        return self.df.loc[self.df["status"] == status]

    def filter_feature(self, feature):
        return self.df[feature]

    def display(self, df):
        print(df.to_string())


data_all = Data(f"{DATA_PATH}/parkinsons_all")
data_mean = Data(f"{DATA_PATH}/parkinsons_mean")
