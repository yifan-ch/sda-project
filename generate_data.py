"""
Generates csv tables with new variables and writes them to disk.
"""

from tools.data_tools import read, write
from pathlib import Path
from env import PATHS


def generate_data():
    """Generate custom csv data from the original"""
    # Drop name column, as they don't contain important info
    df = read("parkinsons.csv", original=True, sep=",")
    df2 = read("parkinsons_subset.csv", original=True, sep=";")

    write(df, "parkinsons_raw.csv")

    # Take the mean of the experiments (as there are multiple measurements per participant)
    grouped_df = df.groupby("id", as_index=False).mean().drop(["id"], axis=1)
    write(grouped_df, "parkinsons_mean.csv")
    grouped_df2 = df2.groupby("id", as_index=False).mean().drop(["id"], axis=1)

    # Calculate the mean of the means and standard deviation
    grouped_df_mean = grouped_df.mean()
    grouped_df_mean_df = grouped_df_mean.to_frame(name="mean").reset_index()
    grouped_df_mean_df["standard_deviation"] = grouped_df.std().values
    write(grouped_df_mean_df, "parkinsons_mean_std.csv")

    # Calculate z-scores for all columns except 'status'
    features = grouped_df.drop(columns=["status"])
    z_scores = (features - features.mean()) / features.std()
    features2 = grouped_df2.drop(columns=["status"])
    z_scores2 = (features2 - features2.mean()) / features2.std()

    # Add unmodified 'status' column back
    z_scores["status"] = grouped_df["status"]
    z_scores2["status"] = grouped_df2["status"]

    # Save the z-scores to a new CSV file
    write(z_scores, "parkinsons_z_scores.csv")
    write(z_scores2, "parkinsons_subset_z_scores.csv")


if __name__ == "__main__":
    # Create path if it doesn't exist
    Path(PATHS["data"]["generated"]).mkdir(parents=True, exist_ok=True)
    generate_data()
