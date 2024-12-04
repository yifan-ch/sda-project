import numpy as np
import pandas as pd
from pathlib import Path
from env import DATA_PATH, DATA_GENERATED_PATH


def write(df, filename):
    df.to_csv(Path(DATA_GENERATED_PATH) / filename, index=False)


# Generate custom csv data from the original
def generate_data():
    # If path doesnt exist, create all missing folders
    Path(DATA_GENERATED_PATH).mkdir(parents=True, exist_ok=True)

    # Drop name column, as they don't contain important info
    df = pd.read_csv(Path(DATA_PATH) / "parkinsons_original.csv").drop("name", axis=1)

    # Create subject id and experiment number columns for easier indexing.
    df.insert(0, "experiment", np.tile(np.arange(0, 6), len(df) // 6))
    df.insert(0, "id", np.repeat(np.arange(0, len(df) // 6), 6))
    # Add variables vocal range and relative pitch
    df.insert(5, "vocal_range", df["MDVP:Fhi(Hz)"] - df["MDVP:Flo(Hz)"])
    df.insert(
        6,
        "relative_avg_pitch",
        ((df["MDVP:Fo(Hz)"] - df["MDVP:Flo(Hz)"]) / (df["MDVP:Fhi(Hz)"] - df["MDVP:Flo(Hz)"])),
    )
    write(df, "parkinsons_raw.csv")

    # Take the mean of the experiments
    grouped_df = df.groupby("id", as_index=False).mean()
    write(grouped_df, "parkinsons_mean.csv")

    # Calculate the mean of the means and standard deviation
    grouped_df_mean = grouped_df.mean()
    grouped_df_mean_df = grouped_df_mean.to_frame(name="mean").reset_index()
    grouped_df_mean_df["standard_deviation"] = grouped_df.std().values
    write(grouped_df_mean_df, "parkinsons_mean_mean.csv")

    # Calculate the overall mean and standard deviation of the means
    mean_of_means = grouped_df.mean()
    std_of_means = grouped_df.std()

    # Calculate z-scores for each person
    z_scores = (grouped_df - mean_of_means) / std_of_means

    # Save the z-scores to a new CSV file
    write(z_scores.drop(["id", "experiment"], axis=1), "parkinsons_z_scores.csv")
    return z_scores


if __name__ == "__main__":
    generate_data()
