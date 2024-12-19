"""Functions to used for data: reading data, writing data into csv file, filtering by status,
calculating z-scores and displaying data."""


import pandas as pd
from pathlib import Path
from env import PATHS


def read(filename, original=False, sep=","):
    """
    Read a CSV file from the specified path (original or generated).

    Parameters:
    filename (str): The name of the file to read.
    original (bool): Whether to read from the original data path. Defaults to False.
    sep (str): The delimiter to use. Defaults to ','.

    Returns:
    DataFrame: The read pandas DataFrame.
    """
    return pd.read_csv(PATHS["data"]["original" if original else "generated"] / filename, sep=sep)


def write(df, filename):
    """
    Write the pandas DataFrame to a CSV file.

    Parameters:
    df (DataFrame): The DataFrame to write.
    filename (str): The name of the file to write.
    """
    Path(PATHS["data"]["generated"]).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(PATHS["data"]["generated"]) / filename, index=False)


def status(df, stat):
    """
    Filter the pandas DataFrame by the specified status.

    Parameters:
    df (DataFrame): The DataFrame to filter.
    stat (str): The status to filter by.

    Returns:
    DataFrame: The filtered DataFrame.
    """
    return df.loc[df["status"] == stat]


def display(df):
    """
    Display the pandas DataFrame as a string.

    Parameters:
    df (DataFrame): The DataFrame to display.
    """
    print(df.to_string())


def df_z_scores():
    """
    Read and return the z-scores DataFrame from a CSV file.

    Returns:
    DataFrame: The z-scores DataFrame.
    """
    df = read("parkinsons_z_scores.csv")
    return df


def df_subset_z_scores():
    """
    Read and return the subset z-scores DataFrame from a CSV file.

    Returns:
    DataFrame: The subset z-scores DataFrame.
    """
    df = read("parkinsons_subset_z_scores.csv")
    return df
