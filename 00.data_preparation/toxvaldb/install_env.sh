#!/bin/bash

# Environment for data preparation and cleaning
uv venv env-data-prep --python=python3.12
source env-data-prep/bin/activate
uv pip install --no-cache -r requirements.txt
python -m ipykernel install --user --name=env-data-prep --display-name "Python (env-data-prep)"

echo "To activate later: source env-data-prep/bin/activate"
echo "To install a package: uv pip install pkg-name"
echo "To update requirements.txt after installing packages: uv pip freeze > requirements.txt"
