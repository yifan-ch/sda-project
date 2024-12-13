from pathlib import Path

DATA_PATH = Path("data")
RESULTS_PATH = Path("results")


PATHS = {
    "data": {
        "original": DATA_PATH / "original",
        "generated": DATA_PATH / "generated",
    },
    "results": {
        "histogram": RESULTS_PATH / "histogram",
        "multiple-regression": RESULTS_PATH / "multiple-regression",
        "logistic-regression": RESULTS_PATH / "logistic-regression",
        "elasticnet-regression": RESULTS_PATH / "elasticnet-regression",
        "vif": RESULTS_PATH / "vif",
    },
}
