"""
Generates csv tables with new variables and writes them to disk.

This script reads original Parkinson's disease datasets, processes them to generate
new datase
ts with calculated statistics, and writes the results to CSV files.
"""

from tools.data_tools import read, write
from pathlib import Path
from env import PATHS


def generate_data():
    """
    Generate custom csv data from the original datasets.

    This function reads the original data files, processes them to calculate
    mean, standard deviation, and z-scores, and writes the results to new CSV files.
    """
    # Read the original data files
    df = read("parkinsons.csv", original=True, sep=",")
    df2 = read("parkinsons_subset.csv", original=True, sep=";")

    # Write the raw data to a new CSV file
    write(df, "parkinsons_raw.csv")

    # Group by 'id' and calculate the mean for each participant, then drop the 'id' column
    grouped_df = df.groupby("id", as_index=False).mean().drop(["id"], axis=1)
    write(grouped_df, "parkinsons_mean.csv")
    grouped_df2 = df2.groupby("id", as_index=False).mean().drop(["id"], axis=1)

    # Calculate the mean and standard deviation for each column
    grouped_df_mean = grouped_df.mean()
    grouped_df_mean_df = grouped_df_mean.to_frame(name="mean").reset_index()
    grouped_df_mean_df["standard_deviation"] = grouped_df.std().values
    write(grouped_df_mean_df, "parkinsons_mean_std.csv")

    # Calculate z-scores for all columns except 'status'
    features = grouped_df.drop(columns=["status"])
    z_scores = (features - features.mean()) / features.std()
    features2 = grouped_df2.drop(columns=["status"])
    z_scores2 = (features2 - features2.mean()) / features2.std()

    # Add unmodified 'status' column back to the z-scores DataFrame
    z_scores["status"] = grouped_df["status"]
    z_scores2["status"] = grouped_df2["status"]

    # Save the z-scores to new CSV files
    write(z_scores, "parkinsons_z_scores.csv")
    write(z_scores2, "parkinsons_subset_z_scores.csv")


if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    Path(PATHS["data"]["generated"]).mkdir(parents=True, exist_ok=True)
    generate_data()
