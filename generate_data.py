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
    write(df, "parkinsons_raw.csv")

    # Take the mean of the experiments
    df.groupby("id", as_index=False).mean()
    write(df, "parkinsons_mean.csv")


if __name__ == "__main__":
    generate_data()
