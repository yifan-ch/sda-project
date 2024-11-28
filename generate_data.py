import numpy as np
import pandas as pd
from pathlib import Path

from env import DATA_PATH

DATA_OUTPUT_PATH = f"{DATA_PATH}/generated"


def write(df, filename):
    Path(DATA_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{DATA_OUTPUT_PATH}/{filename}.csv", index=False)


# drop name column, as they don't contain important info
df = pd.read_csv(f"{DATA_PATH}/parkinsons_original.csv").drop("name", axis=1)

# Create subject id and experiment number columns for easier indexing.
df.insert(0, "experiment", np.tile(np.arange(0, 6), len(df) // 6))
df.insert(0, "id", np.repeat(np.arange(0, len(df) // 6), 6))
write(df, "parkinsons_raw")

# Take the mean of the experiments
df = df.groupby("id", as_index=False).mean()
write(df, "parkinsons_mean")
