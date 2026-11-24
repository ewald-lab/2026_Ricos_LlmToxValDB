#!/bin/bash

# Environment for text-to-onto processing
# uv-managed Python: compute nodes do not have the login node's /usr/bin/python3.12
uv venv env-text-to-onto --python=3.12 --managed-python
source env-text-to-onto/bin/activate
uv pip install --no-cache -r requirements.txt
python -m ipykernel install --user --name=env-text-to-onto --display-name "Python (env-text-to-onto)"

echo "To activate later: source env-text-to-onto/bin/activate"
echo "To install a package: uv pip install pkg-name"
echo "To update requirements.txt after installing packages: uv pip freeze > requirements.txt"
