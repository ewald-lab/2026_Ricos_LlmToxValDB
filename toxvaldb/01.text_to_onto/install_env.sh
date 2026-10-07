#!/bin/bash

# Environment for text-to-onto processing
uv venv env-text-to-onto --python=python3.12
source env-text-to-onto/bin/activate
uv pip install --no-cache -r requirements.txt
python -m ipykernel install --user --name=env-text-to-onto --display-name "Python (env-text-to-onto)"

echo "To activate later: source env-text-to-onto/bin/activate"
echo "To install a package: uv pip install pkg-name"
echo "To update requirements.txt after installing packages: uv pip freeze > requirements.txt"
