#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH -J "prepare_guidelines"
#SBATCH -o "logs/prepare_guidelines_%j.out"
#SBATCH -e "logs/prepare_guidelines_%j.err"

set -euo pipefail

# --- Configuration ---
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)
    SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# --- Arguments ---
INPUT_DIR="${1:-${REPO_ROOT}/outputs/endpoints}"
OUTPUT_DIR="${2:-${REPO_ROOT}/outputs/normalized_endpoints}"

# --- Environment Setup ---
if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/env-data-prep/bin/activate"
fi

echo "=== Prepare Guidelines Job ==="

python "${REPO_ROOT}/scripts/00.prepare_guidelines.py" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "Guideline preparation completed."
