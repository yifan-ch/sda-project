# Vocal analysis as a screening for Parkinson’s disease
Parkinson’s disease is a neurodegenerative disease. It is related to the death of dopamine producing cells in a part of the brain called the substantia nigra. This results in too little dopamine being available in the brain, which leads to several symptoms. The most obvious are the motoric symptoms such as tremors and rigidity, but there are also cognitive symptoms, particularly later in the disease progression. One perhaps lesser known symptom of parkinson’s disease is that it can affect your voice. This is what we wanted to research. Can we, by way of measuring vocal characteristics, screen for PD to more easily and cheaply allocate care to people that might need it?

By Lynne Vogel, Mink van Maanen & Yifan Chen.

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
./venv/Scripts/activate.bat # windows
source venv/bin/activate    # mac
```

Lastly install the required dependencies:
```bash
pip install -r requirements.txt
```

If you wish to recompile depencencies, `pip-tools`, `pipreqs` and `Make` must be installed:
```bash
# install (only do this once)
pip install pip-tools
pip install pipreqs

# compile
Make
```

## Usage
Generate the results by running `python generate_results.py`. The generated results can be found in `./results`
Running this will also generate modified datasets from the original dataset(s), which can be found in `./data/generated`

Optionally it's possible to use command-line arguments to modify the tests.
For more info, run `python generate_results.py -h`.
