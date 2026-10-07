#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH -J "normalize_guidelines"
#SBATCH -o "logs/prepare_guidelines_%j.out"
#SBATCH -e "logs/prepare_guidelines_%j.err"

if [ -n "$SLURM_JOB_ID" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${SCRIPT_DIR}/logs"

INPUT_DIR="${1:-${REPO_ROOT}/outputs}"
OUTPUT_DIR="${2:-${REPO_ROOT}/outputs}"

if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/env-data-prep/bin/activate"
elif [ -f "${PREP_ROOT}/../env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/../env-data-prep/bin/activate"
fi
fi

echo "=========================================================="
echo "=== Normalize Guidelines Job ==="
echo "Input Dir  : ${INPUT_DIR}"
echo "Output Dir : ${OUTPUT_DIR}"
echo "=========================================================="

python "${REPO_ROOT}/scripts/01.normalize_guidelines.py" \
    --input_dir  "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "Guideline preparation completed."
