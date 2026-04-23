#!/bin/bash

# Environment for use cases analysis
uv venv env-use-cases --python=python3.12
source env-use-cases/bin/activate
uv pip install --no-cache -r requirements.txt
python -m ipykernel install --user --name=env-use-cases --display-name "Python (env-use-cases)"

echo "To activate later: source env-use-cases/bin/activate"
echo "To install a package: uv pip install pkg-name"
echo "To update requirements.txt after installing packages: uv pip freeze > requirements.txt"
