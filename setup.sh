#!/bin/bash

venv_dir="venv"

if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install it before proceeding."
    exit 1
fi

if [ ! -d "$venv_dir" ]; then
    echo "Creating virtual environment in '$venv_dir'..."
    python3 -m venv "$venv_dir"
else
    echo "The virtual environment '$venv_dir' already exists."
fi

source "$venv_dir/bin/activate"

if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "No requirements.txt file found."
fi

echo "Virtual environment setup completed successfully."

