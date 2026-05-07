#!/bin/bash

# Environment for evaluation and RAG analysis
uv venv env-evaluation --python=python3.12
source env-evaluation/bin/activate
uv pip install --no-cache -r requirements.txt
python -m ipykernel install --user --name=env-evaluation --display-name "Python (env-evaluation)"

echo "To activate later: source env-evaluation/bin/activate"
echo "To install a package: uv pip install pkg-name"
echo "To update requirements.txt after installing packages: uv pip freeze > requirements.txt"
