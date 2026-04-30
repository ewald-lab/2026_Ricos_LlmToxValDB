#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH -J "merge_llm"
#SBATCH -o "logs/merge_llm%j.out"
#SBATCH -e "logs/merge_llm%j.err"

# --- Configuration ---
# Get script directory (works when submitted via sbatch or run locally)
if [ -n "$SLURM_JOB_ID" ]; then
    SCRIPT_PATH=$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)
    SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

POS_FILE="${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_02.tsv"
NEG_FILE="${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_02_negatives.tsv"
OUTPUT_FILE="${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_03_merged.tsv"

# --- Environment Setup ---
source "${PREP_ROOT}/env-data-prep/bin/activate"

echo "=== Merging Positive and Negative Toxicological Effects ==="
echo "Positives: ${POS_FILE}"
echo "Negatives: ${NEG_FILE}"

python "${REPO_ROOT}/scripts/02.merge_pos_neg.py" \
    --positives "${POS_FILE}" \
    --negatives "${NEG_FILE}" \
    --output "${OUTPUT_FILE}"

if [ $? -ne 0 ]; then
    echo "Error: Merging failed."
    exit 1
fi

echo "Merging completed successfully: ${OUTPUT_FILE}"
