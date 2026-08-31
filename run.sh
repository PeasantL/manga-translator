#!/bin/bash

# Check if the activate script exists
if [ -f "./.venv/bin/activate" ]; then
    echo "Activating existing virtual environment."
    source "./.venv/bin/activate"
else
    echo "Virtual environment does not exist. Creating virtual environment."
    python3 -m venv .venv
    source "./.venv/bin/activate"
    pip install -r requirements.txt  
fi

# The model weights are not in the repository.
if [ ! -f "./models/detection.pt" ]; then
    echo "Model weights are missing. Run ./fetch_models.sh first."
    exit 1
fi

# Each folder in input/ is one chapter. main.py takes chapters, not a folder of
# them, so expand them here rather than making it guess which it was handed.
shopt -s nullglob
chapters=(input/*/)

if [ ${#chapters[@]} -eq 0 ]; then
    echo "No chapters in input/. Put each chapter's pages in a folder there."
    exit 1
fi

python3 main.py -f "${chapters[@]}"
