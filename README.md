# AYS-Consistency-Models-MNIST
Repository for Final Project for the Generative AI Class at WPI

## Setup Instructions

### Installing UV (Ultraviolet)
This project uses `uv` for python project dependency management. Please install `uv` from here, depending on your OS: https://docs.astral.sh/uv/getting-started/installation/


### Activating Existing UV Environment
Once done with `uv` installation, please run the following commmand to activate `uv` env:

ONE-TIME-ONLY:
```bash
# Navigate to the project folder
cd /path/to/CS-552-Generative-AI-Final-Project

# Create existing uv env using pyproject.toml and uv.lock
uv sync
```

To activate virtual env:
```bash
# Active the existing virtual env
source .venv/bin/activate
```
