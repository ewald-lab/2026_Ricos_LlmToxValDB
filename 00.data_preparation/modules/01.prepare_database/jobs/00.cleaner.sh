#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH -J "clean_base"
#SBATCH -o "logs/clean_base%j.out"
#SBATCH -e "logs/clean_base%j.err"

# --- Configuration ---
SCRIPT_PATH=$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# --- Arguments ---
INPUT_DB="${1:-${PREP_ROOT}/modules/00.dowload_data/outputs/ToxValDB_v9.7.0_Unified.tsv}"
OUTPUT_DB="${2:-${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_00.tsv}"

# --- Environment Setup ---
source "${PREP_ROOT}/env-data-prep/bin/activate"

echo "=== Base Cleaning ==="
echo "Input: ${INPUT_DB}"
echo "Output: ${OUTPUT_DB}"

python "${REPO_ROOT}/scripts/00.clean.py" --input "${INPUT_DB}" --output "${OUTPUT_DB}"

if [ $? -ne 0 ]; then
    echo "Error: Base cleaning failed."
    exit 1
fi
echo "Base cleaning completed."
