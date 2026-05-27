#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH -J "eval_summary"
#SBATCH -o "logs/eval_summary_%j.out"
#SBATCH -e "logs/eval_summary_%j.err"

# Ensure logs directory exists (run from jobs/ dir)
mkdir -p logs

# Resolve paths relative to this script's location
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
if [ -z "$SCRIPT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
MODULE_DIR="$(dirname "${SCRIPT_DIR}")"
EVAL_DIR="$(dirname "$(dirname "${MODULE_DIR}")")"

EXPERT_EXPANDED_DIR="${EVAL_DIR}/modules/01.eval_experts/outputs"
LLM_EXPANDED_DIR="${MODULE_DIR}/outputs"
OUTPUT_DIR="${MODULE_DIR}/outputs/figures"

# Activate environment
source "${EVAL_DIR}/env-evaluation/bin/activate"

echo "=== Generating Summary Table and Plot ==="
echo "  Expert Expanded dir: ${EXPERT_EXPANDED_DIR}"
echo "  LLM Expanded dir   : ${LLM_EXPANDED_DIR}"
echo "  Figures dir        : ${OUTPUT_DIR}"

python3 "${MODULE_DIR}/scripts/06.summary.py" \
    --expert_expanded_dir "${EXPERT_EXPANDED_DIR}" \
    --llm_expanded_dir    "${LLM_EXPANDED_DIR}" \
    --output_dir           "${OUTPUT_DIR}"

echo "Done. Summary plot written to: ${OUTPUT_DIR}"
