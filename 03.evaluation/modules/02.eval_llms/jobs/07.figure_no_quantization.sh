#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "fig_no_quant"
#SBATCH -o "logs/fig_no_quant_%j.out"
#SBATCH -e "logs/fig_no_quant_%j.err"

# Ensure logs directory exists (run from jobs/ dir)
mkdir -p logs

# Resolve paths relative to this script's location
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
if [ -z "$SCRIPT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
MODULE_DIR="$(dirname "${SCRIPT_DIR}")"
EVAL_DIR="$(dirname "$(dirname "${MODULE_DIR}")")"

# Configuration
ONTOLOGY="mp"
EXPERT_EXPANDED_DIR="${EVAL_DIR}/modules/01.eval_experts/outputs"
EXPERT_ORIG_DIR="${EVAL_DIR}/modules/01.eval_experts/inputs/${ONTOLOGY}"
LLM_EXPANDED_DIR="${MODULE_DIR}/outputs"
OUTPUT_DIR="${MODULE_DIR}/outputs/figures"

# Activate environment
source "${EVAL_DIR}/env-evaluation/bin/activate"

echo "=== Generating Quantization Comparison Table ==="
echo "  Ontology           : ${ONTOLOGY}"
echo "  Expert Expanded dir: ${EXPERT_EXPANDED_DIR}"
echo "  LLM Expanded dir   : ${LLM_EXPANDED_DIR}"
echo "  Figures dir        : ${OUTPUT_DIR}"

python3 "${MODULE_DIR}/scripts/07.figure_no_quantization.py" \
    --expert_expanded_dir "${EXPERT_EXPANDED_DIR}" \
    --expert_orig_dir     "${EXPERT_ORIG_DIR}" \
    --llm_expanded_dir    "${LLM_EXPANDED_DIR}" \
    --output_dir           "${OUTPUT_DIR}" \
    --ontology             "${ONTOLOGY}"

echo "Done. Table written to: ${OUTPUT_DIR}/quantization_comparison_table.png"
