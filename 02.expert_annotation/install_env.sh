#!/bin/bash

# Environment for expert annotation test generation
uv venv env-expert-annot --python=python3.12
source env-expert-annot/bin/activate
uv pip install --no-cache -r requirements.txt
python -m ipykernel install --user --name=env-expert-annot --display-name "Python (env-expert-annot)"

echo "To activate later: source env-expert-annot/bin/activate"
echo "To install a package: uv pip install pkg-name"
echo "To update requirements.txt after installing packages: uv pip freeze > requirements.txt"
