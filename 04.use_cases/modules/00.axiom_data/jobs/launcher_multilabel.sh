#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -J "toxval_multilabel_classification"
#SBATCH -o "logs/out_multilabel_%j.out"
#SBATCH -e "logs/err_multilabel_%j.err"

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
OUTPUT_FILE="${OUTPUT_DIR}/multilabel_results.parquet"

# Environment setup
module purge
module load cuda/12.8.0
module load gcc

# Load virtual environment
source "${PROJECT_ROOT}/env-use-cases/bin/activate"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run multi-label classification
python3 "${MODULE_ROOT}/scripts/classify_multilabel.py" \
    --features_path "${FEATURES_PARQUET}" \
    --labels_path "${LABELS_CSV}" \
    --mapping_path "${MAPPING_CSV}" \
    --output "${OUTPUT_FILE}" \
    --n_repeats 1 \
    --n_splits 5

echo "Multilabel classification job completed."
