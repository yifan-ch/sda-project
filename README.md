# SDA Project
SDA Project by Lynne Vogel, Mink van Maanen & Yifan Chen

## Installation
Requirements:
* Python 3.11

First, create a virtual environment:
```bash
python3 -m venv venv
```

Activate it using the appropriate command for your platform:
```bash
bash venv/bin/activate      # linux
./venv/Scripts/activate.bat   # windows
source venv/bin/activate    # mac
```

Lastly install the required dependencies:
```bash
pip install -r requirements.txt
```

To recompile depencencies, `pip-tools`, `pipreqs` and `Make` must be installed:
```bash
# install (only do this once)
pip install pip-tools
pip install pipreqs

# compile
Make

```

## Usage
1. Generate the data by running `python3 generate_data.py`. The generated data can be found in `./data/generated/`.
2. Generate the results by running `python3 generate_results.py`. The generated results can be found in `./results/`
