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
./venv/Scripts/Activate.bat   # windows
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
