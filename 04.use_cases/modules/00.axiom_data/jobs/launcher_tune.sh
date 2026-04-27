#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -J "tox_tuner_optuna"
#SBATCH -o "logs/out_tune_%j.out"
#SBATCH -e "logs/err_tune_%j.err"

# Path discovery
JOBS_DIR="$SLURM_SUBMIT_DIR"
MODULE_ROOT="$(cd "${JOBS_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"

# Input features and labels
FEATURES_PARQUET="${MODULE_ROOT}/inputs/level6_axiom.parquet"
LABELS_CSV="${MODULE_ROOT}/outputs/ToxValDB_Grouped_Filtered_5pc.csv"
MAPPING_CSV="${MODULE_ROOT}/inputs/name_to_dtxsid_mapping.csv"

# Output results
OUTPUT_DIR="${MODULE_ROOT}/outputs"
OUTPUT_FILE="${OUTPUT_DIR}/multilabel_results_tuned.parquet"

# Environment setup
module purge
module load cuda/12.8.0
module load gcc

# Load virtual environment
source "${PROJECT_ROOT}/env-use-cases/bin/activate"

echo "Starting Hyperparameter Tuning (Winner-Stays logic with Optuna)..."
python3 "${MODULE_ROOT}/scripts/tune_multilabel.py" \
    --features_path "${FEATURES_PARQUET}" \
    --labels_path "${LABELS_CSV}" \
    --mapping_path "${MAPPING_CSV}" \
    --output "${OUTPUT_FILE}" \
    --n_splits 5

echo "Tuning finished. Generating Dashboard..."
python3 "${MODULE_ROOT}/scripts/analyze_results.py" \
    --input "${OUTPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --suffix "_tuned"

echo "Process complete."
