#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH -J "toxval_analyze_results"
#SBATCH -o "logs/out_analysis_%j.out"
#SBATCH -e "logs/err_analysis_%j.err"

# Path discovery
JOBS_DIR="$SLURM_SUBMIT_DIR"
MODULE_ROOT="$(cd "${JOBS_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"

OUTPUT_DIR="${MODULE_ROOT}/outputs"
INPUT_FILE="${OUTPUT_DIR}/multilabel_results.parquet"

# Environment setup
source "${PROJECT_ROOT}/env-use-cases/bin/activate"

# Run analysis
python3 "${MODULE_ROOT}/scripts/analyze_results.py" \
    --input ${INPUT_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --suffix consensus

echo "Analysis completed."
