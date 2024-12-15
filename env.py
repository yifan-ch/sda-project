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
        "multiple-regression-elasticnet": RESULTS_PATH / "multiple-regression-elasticnet",
        "logistic-regression-elasticnet": RESULTS_PATH / "logistic-regression-elasticnet",
        "regressions-combined": RESULTS_PATH / "regressions-combined",
        "vif": RESULTS_PATH / "vif",
    },
}
