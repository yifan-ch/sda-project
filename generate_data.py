from data_model import read, write
from pathlib import Path
from env import PATHS


# Generate custom csv data from the original
def generate_data():

    # Drop name column, as they don't contain important info
    df = read("parkinsons.csv", original=True, sep=",")

    write(df, "parkinsons_raw.csv")

    # Take the mean of the experiments
    grouped_df = df.groupby("id", as_index=False).mean().drop(["id"], axis=1)
    write(grouped_df, "parkinsons_mean.csv")

    # Calculate the mean of the means and standard deviation
    grouped_df_mean = grouped_df.mean()
    grouped_df_mean_df = grouped_df_mean.to_frame(name="mean").reset_index()
    grouped_df_mean_df["standard_deviation"] = grouped_df.std().values
    write(grouped_df_mean_df, "parkinsons_mean_std.csv")

    # Calculate the overall mean and standard deviation of the means
    # mean_of_means = grouped_df.mean()
    # std_of_means = grouped_df.std()

    # Calculate z-scores for all columns except 'status'
    features = grouped_df.drop(columns=["status"])
    z_scores = (features - features.mean()) / features.std()

    # Add the unmodified 'status' column back
    z_scores["status"] = grouped_df["status"]

    # Save the z-scores to a new CSV file
    write(z_scores, "parkinsons_z_scores.csv")


if __name__ == "__main__":
    Path(PATHS["data"]["generated"]).mkdir(parents=True, exist_ok=True)

    generate_data()
