#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "eval_llm_responses"
#SBATCH -o "logs/eval_llm_responses_%j.out"
#SBATCH -e "logs/eval_llm_responses_%j.err"

# Ensure logs directory exists (run from jobs/ dir)
mkdir -p logs

# Resolve paths relative to this script's location
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
MODULE_DIR="$(dirname "${SCRIPT_DIR}")"
EVAL_DIR="$(dirname "$(dirname "${MODULE_DIR}")")"

# ─── Configuration ────────────────────────────────────────────────────────────
ONTOLOGY="mp"
ROOT_TERM="mammalian phenotype"
# ──────────────────────────────────────────────────────────────────────────────

EXPANDED_DIR="${MODULE_DIR}/outputs"
ORIG_DIR="${MODULE_DIR}/outputs"
EXPERT_EXPANDED_DIR="${EVAL_DIR}/modules/01.eval_experts/outputs"
EXPERT_ORIG_DIR="${EVAL_DIR}/modules/01.eval_experts/inputs/${ONTOLOGY}"
OWL_FILE="${EVAL_DIR}/../00.data_preparation/modules/02.prepare_ontologies/inputs/owl_files/${ONTOLOGY}.ttl"
OUTPUT_DIR="${MODULE_DIR}/outputs/figures_llm_only"

# Activate environment
source "${EVAL_DIR}/env-evaluation/bin/activate"

echo "=== Evaluating LLM responses ==="
echo "  Ontology     : ${ONTOLOGY}"
echo "  OWL file     : ${OWL_FILE}"
echo "  Expanded dir : ${EXPANDED_DIR}"
echo "  Original dir : ${ORIG_DIR}"
echo "  Figures dir  : ${OUTPUT_DIR}"

python3 "${MODULE_DIR}/scripts/03.generate_llm_plots.py" \
    --expanded_dir        "${EXPANDED_DIR}" \
    --orig_dir            "${ORIG_DIR}" \
    --owl_file            "${OWL_FILE}" \
    --output_dir          "${OUTPUT_DIR}" \
    --ontology            "${ONTOLOGY}" \
    --root_term           "${ROOT_TERM}" \
    --expert_expanded_dir "${EXPERT_EXPANDED_DIR}" \
    --expert_orig_dir     "${EXPERT_ORIG_DIR}"

echo "Done. Plots written to: ${OUTPUT_DIR}"
