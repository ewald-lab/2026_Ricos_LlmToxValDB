#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "toxval_merge_pos_neg"
#SBATCH -o "logs/out_merge_%A.out"
#SBATCH -e "logs/err_merge_%A.err"

SCRIPT_DIR=$(dirname "$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)")
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${PREP_ROOT}/env-data-prep/bin/activate"

python "${REPO_ROOT}/scripts/02.merge_pos_neg.py" \
    --positives "${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_02_positives.tsv" \
    --negatives "${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_02_negatives.tsv" \
    --output    "${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv"

echo "Merge completed."
