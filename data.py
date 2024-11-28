import numpy as np
import pandas as pd


class Data:
    def __init__(self, path="data/parkinsons_data.csv"):
        self.df = pd.read_csv(path).drop("name", axis=1)

    # def
