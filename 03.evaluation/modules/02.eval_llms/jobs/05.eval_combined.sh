#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "eval_combined"
#SBATCH -o "logs/eval_combined_%j.out"
#SBATCH -e "logs/eval_combined_%j.err"

# Ensure logs directory exists (run from jobs/ dir)
mkdir -p logs

# Resolve paths relative to this script's location
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
if [ -z "$SCRIPT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
MODULE_DIR="$(dirname "${SCRIPT_DIR}")"
EVAL_DIR="$(dirname "$(dirname "${MODULE_DIR}")")"

# ─── Configuration ────────────────────────────────────────────────────────────
ONTOLOGY="mp"
ROOT_TERM="mammalian phenotype"
# ──────────────────────────────────────────────────────────────────────────────

EXPERT_EXPANDED_DIR="${EVAL_DIR}/modules/01.eval_experts/outputs"
EXPERT_ORIG_DIR="${EVAL_DIR}/modules/01.eval_experts/inputs/${ONTOLOGY}"
LLM_EXPANDED_DIR="${MODULE_DIR}/outputs"
LLM_ORIG_DIR="${MODULE_DIR}/outputs"

OWL_FILE="${EVAL_DIR}/../00.data_preparation/modules/02.prepare_ontologies/inputs/owl_files/${ONTOLOGY}.ttl"
ONTOLOGY_EMBEDDING_DB="${EVAL_DIR}/../00.data_preparation/modules/02.prepare_ontologies/outputs/kalm/${ONTOLOGY}.db"
OUTPUT_DIR="${MODULE_DIR}/outputs/figures"

# Activate environment
source "${EVAL_DIR}/env-evaluation/bin/activate"

echo "=== Evaluating Combined Responses (Humans + LLMs) ==="
echo "  Ontology           : ${ONTOLOGY}"
echo "  OWL file           : ${OWL_FILE}"
echo "  Expert Expanded dir: ${EXPERT_EXPANDED_DIR}"
echo "  LLM Expanded dir   : ${LLM_EXPANDED_DIR}"
echo "  Figures dir        : ${OUTPUT_DIR}"
echo "  Manual terms       : excluded from evaluation/plots"
echo "  Jaccard boxplot    : skipped"

python3 "${MODULE_DIR}/scripts/05.eval_combined.py" \
    --expert_expanded_dir "${EXPERT_EXPANDED_DIR}" \
    --expert_orig_dir     "${EXPERT_ORIG_DIR}" \
    --llm_expanded_dir    "${LLM_EXPANDED_DIR}" \
    --llm_orig_dir         "${LLM_ORIG_DIR}" \
    --owl_file             "${OWL_FILE}" \
    --output_dir           "${OUTPUT_DIR}" \
    --ontology             "${ONTOLOGY}" \
    --root_term            "${ROOT_TERM}" \
    --ontology_embedding_db "${ONTOLOGY_EMBEDDING_DB}" \
    --exclude_average_human \
    --skip_inter_annotator_agreement_boxplot \
    --expert_equivalence_file "${EVAL_DIR}/modules/01.eval_experts/inputs/expert_equivalence.txt"

echo "Done. Combined plots written to: ${OUTPUT_DIR}"
