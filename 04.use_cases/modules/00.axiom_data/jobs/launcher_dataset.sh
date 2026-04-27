#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "toxval_dataset"
#SBATCH -o "logs/out_dataset_%j.out"
#SBATCH -e "logs/err_dataset_%j.err"

# Path discovery
JOBS_DIR="$SLURM_SUBMIT_DIR"
MODULE_ROOT="$(cd "${JOBS_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"
GIT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

# Data paths
CLASSIFIED_CSV="${GIT_ROOT}/01.text_to_onto/modules/00.text_to_onto/outputs/ToxValDB_Classified_mp_Qwen3.5-122B-A10B-FP8.csv"
GROUPED_CSV="${MODULE_ROOT}/outputs/ToxValDB_Grouped_by_DTXSID_Tiered.csv"
FINAL_CSV="${MODULE_ROOT}/outputs/ToxValDB_Grouped_Filtered_5pc.csv"

# Environment setup
module purge
module load gcc

# Load virtual environment
source "${PROJECT_ROOT}/env-use-cases/bin/activate"

# Execute python scripts
echo "Starting grouping by DTXSID..."
python3 "${MODULE_ROOT}/scripts/group_by_dtxsid_tiered.py" \
    --input "${CLASSIFIED_CSV}" \
    --output "${GROUPED_CSV}"

echo "Starting dataset finalization..."
python3 "${MODULE_ROOT}/scripts/finalize_dataset_tiered.py" \
    --input "${GROUPED_CSV}" \
    --output "${FINAL_CSV}"

echo "Dataset creation completed."
