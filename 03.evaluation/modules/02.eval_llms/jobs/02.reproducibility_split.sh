#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "reproducibility_split"
#SBATCH -o "logs/reproducibility_split_%j.out"
#SBATCH -e "logs/reproducibility_split_%j.err"

# Ensure logs directory exists (run from jobs/ dir)
mkdir -p logs

# Resolve paths relative to this script's location
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
if [ -z "$SCRIPT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
MODULE_DIR="$(dirname "${SCRIPT_DIR}")"
EVAL_DIR="$(dirname "$(dirname "${MODULE_DIR}")")"

# Activate environment
source "${EVAL_DIR}/env-evaluation/bin/activate"

echo "=== Running reproducibility split ==="

python3 "${MODULE_DIR}/scripts/02.reproducibility_split.py"

echo "Done."
