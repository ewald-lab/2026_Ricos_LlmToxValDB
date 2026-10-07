#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "toxval_merge_pos_neg"
#SBATCH -o "logs/out_merge_%A.out"
#SBATCH -e "logs/err_merge_%A.err"

if [ -n "$SLURM_JOB_ID" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${SCRIPT_DIR}/logs"

POSITIVES="${1:-${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_02_positives.tsv}"
NEGATIVES="${2:-${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_02_negatives.tsv}"
OUTPUT="${3:-${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv}"

if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/env-data-prep/bin/activate"
elif [ -f "${PREP_ROOT}/../env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/../env-data-prep/bin/activate"
fi

python "${REPO_ROOT}/scripts/02.merge_pos_neg.py" \
    --positives "${POSITIVES}" \
    --negatives "${NEGATIVES}" \
    --output    "${OUTPUT}"

echo "Merge completed."
