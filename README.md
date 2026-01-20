# 🧪 Ewald Lab Data Analysis Template
This is a template for GitHub repositories for data analysis projects in the Ewald Lab. By sticking to some structure and styling conventions, we make our code more reproducible and reusable. At the repo root, we enforce code style and linting with **Ruff** via pre-commit hooks. See the [Ewald Lab GitHub guidelines](https://ewaldlab.org/handbook/site/git-repo/) for more details.

If you are confused about any of the repository components or instructions, please ask someone to explain!

## Setup Instructions

### 1. Make a new repository based on this template
- Above the file list, click **Use this template** and select **Create a new repository**
- Set the owner to **ewald-lab**
- Name the repository following the YEAR_PERSON_TOPIC format, for example 2025_Ewald_CircadianRhythm
- Set the repository visilibility to private - you can change this to public after discussing with other project members
- Click **Create repository from template**

### 2. Fork the repository
Fork the repository and clone your fork. This makes it easier to manage contributions from multiple collaborators. For more details on how to sync your fork with the main branch, including making contributions, see the documentation [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). 

### 3. Set up development tools
Use uv to install the development environment in the repository root. Installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
uv pip install -r requirements-dev.txt
```
Now install pre-commit hooks:
```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
```

### 4. Set up first analysis module
Each analysis module has its own Python environment managed by [uv](https://github.com/astral-sh/uv). To install and activate a new environment:

```bash
cd ./00.module_name
chmod ugo+x install_env.sh
./install_env.sh
source env-name/bin/activate
```
To install a new package with uv: `uv pip install pkg-name`. To add newly installed packages to the dependency file: `uv pip freeze > requirements.txt`.

### 5. Modify the README
Replace the template README introduction and setup instructions with:
- 1-2 sentences background on the project
- Specific hypotheses you are testing or engineering problems you are trying to solve
- Brief description of each section of the analysis






