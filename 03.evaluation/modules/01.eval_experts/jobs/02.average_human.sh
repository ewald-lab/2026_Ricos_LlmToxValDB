#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH -J "average_human"
#SBATCH -o "logs/average_human_%j.out"
#SBATCH -e "logs/average_human_%j.err"

# Ensure logs directory exists (run from jobs/ dir)
mkdir -p logs

# Resolve paths relative to this script's location
# (Handles Slurm script spooling by falling back to SLURM_SUBMIT_DIR)
if [[ "${BASH_SOURCE[0]}" == /var/spool/slurmd/* ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

MODULE_DIR="$(dirname "${SCRIPT_DIR}")"
EVAL_DIR="$(dirname "$(dirname "${MODULE_DIR}")")"

INPUT_DIR="${MODULE_DIR}/outputs"
OUTPUT_FILE="${MODULE_DIR}/outputs/Expanded_average_human.csv"

# Activate environment
source "${EVAL_DIR}/env-evaluation/bin/activate"

echo "=== Creating average human profile ==="
echo "  Input dir  : ${INPUT_DIR}"
echo "  Output file: ${OUTPUT_FILE}"

python3 "${MODULE_DIR}/scripts/average_human.py" \
    --input_dir   "${INPUT_DIR}" \
    --output_file "${OUTPUT_FILE}" \
    --min_experts 2

echo "Done."
