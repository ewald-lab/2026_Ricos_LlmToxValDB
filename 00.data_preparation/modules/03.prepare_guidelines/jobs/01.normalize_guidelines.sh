#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH -J "prepare_guidelines"
#SBATCH -o "logs/prepare_guidelines_%j.out"
#SBATCH -e "logs/prepare_guidelines_%j.err"

SCRIPT_DIR=$(dirname "$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)")
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${PREP_ROOT}/env-data-prep/bin/activate"

echo "=== Normalize Guidelines Job ==="
python "${REPO_ROOT}/scripts/01.normalize_guidelines.py" \
    --input_dir  "${1:-${REPO_ROOT}/outputs}" \
    --output_dir "${2:-${REPO_ROOT}/outputs}"

echo "Guideline preparation completed."
