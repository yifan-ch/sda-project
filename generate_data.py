import numpy as np
import data_model


# Generate custom csv data from the original
def generate_data():

    # Drop name column, as they don't contain important info
    df = data_model.read("parkinsons2.csv", original=True, sep=";")

    data_model.write(df, "parkinsons2_raw.csv")

    # Take the mean of the experiments
    grouped_df = df.groupby("id", as_index=False).mean().drop(["id"], axis=1)
    data_model.write(grouped_df, "parkinsons2_mean.csv")

    # Calculate the mean of the means and standard deviation
    grouped_df_mean = grouped_df.mean()
    grouped_df_mean_df = grouped_df_mean.to_frame(name="mean").reset_index()
    grouped_df_mean_df["standard_deviation"] = grouped_df.std().values
    data_model.write(grouped_df_mean_df, "parkinsons2_mean_std.csv")

    # Calculate the overall mean and standard deviation of the means
    mean_of_means = grouped_df.mean()
    std_of_means = grouped_df.std()

    # Calculate z-scores for each person
    z_scores = (grouped_df - mean_of_means) / std_of_means

    # Save the z-scores to a new CSV file
    data_model.write(z_scores, "parkinsons2_z_scores.csv")


if __name__ == "__main__":
    generate_data()
