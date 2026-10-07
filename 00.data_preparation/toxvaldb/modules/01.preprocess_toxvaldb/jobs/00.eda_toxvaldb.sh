#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "toxval_eda"
#SBATCH -o "logs/out_eda_%A.out"
#SBATCH -e "logs/err_eda_%A.err"

if [ -n "$SLURM_JOB_ID" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${REPO_ROOT}/outputs"

NOTEBOOK="${1:-00_eda_toxvaldb.ipynb}"

if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/env-data-prep/bin/activate"
elif [ -f "${PREP_ROOT}/../env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/../env-data-prep/bin/activate"
fi
fi

cd "${REPO_ROOT}/analysis"

echo "Executing notebook: ${NOTEBOOK} ..."
jupyter nbconvert --to notebook --execute "${NOTEBOOK}" --inplace

echo "Notebook execution completed."
