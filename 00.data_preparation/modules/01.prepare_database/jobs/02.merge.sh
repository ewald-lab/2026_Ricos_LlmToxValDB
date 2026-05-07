#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "toxval_merge_pos_neg"
#SBATCH -o "logs/out_merge_%A.out"
#SBATCH -e "logs/err_merge_%A.err"

SCRIPT_PATH=$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# --- Environment Setup ---
source "${PREP_ROOT}/env-data-prep/bin/activate"

# Add scripts directory to PYTHONPATH
export PYTHONPATH="${REPO_ROOT}/scripts:$PYTHONPATH"

POSIX_FILE="${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_02_positives.tsv"
NEGAT_FILE="${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_02_negatives.tsv"
OUTPUT_FILE="${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv"

echo "=========================================================="
echo "Merging Positives and Negatives"
echo "Positives: ${POSIX_FILE}"
echo "Negatives: ${NEGAT_FILE}"
echo "Output:    ${OUTPUT_FILE}"
echo "=========================================================="

python "${REPO_ROOT}/scripts/02.merge_pos_neg.py" \
    --positives "${POSIX_FILE}" \
    --negatives "${NEGAT_FILE}" \
    --output "${OUTPUT_FILE}"

echo "Merge completed."
