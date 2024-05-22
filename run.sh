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

# Run the inference script
python3 main.py -f input