import pandas as pd
from pathlib import Path
from env import PATHS


def read(filename, original=False, sep=","):
    # Read csv file
    return pd.read_csv(PATHS["data"]["original" if original else "generated"] / filename, sep=sep)

def write(df, filename):
    # If path doesnt exist, create all missing folders
    Path(PATHS["data"]["generated"]).mkdir(parents=True, exist_ok=True)
    # Write pd dataframe to csv file
    df.to_csv(Path(PATHS["data"]["generated"]) / filename, index=False)

def status(df, stat):
    # Filter pd dataframe by status
    return df.loc[df["status"] == stat]

def display(df):
    # Display pd dataframe
    print(df.to_string())

def df_z_scores():
    # Return z-scores as a pd dataframe
    df = read("parkinsons_z_scores.csv")
    return df
