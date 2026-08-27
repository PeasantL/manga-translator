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
    echo "Model weights are missing. Run ./scripts/fetch_models.sh first."
    exit 1
fi

# Run the inference script
python3 main.py -f input
