#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "eval_combined"
#SBATCH -o "logs/eval_combined_%j.out"
#SBATCH -e "logs/eval_combined_%j.err"

# Resolve paths relative to this script location
if [[ "${BASH_SOURCE[0]}" == /var/spool/slurmd/* ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"
TOXVALDB_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${TOXVALDB_ROOT}/.." && pwd)"

mkdir -p "${SCRIPT_DIR}/logs" "${MODULE_ROOT}/outputs/figures/decisions"

# ─── Configuration ────────────────────────────────────────────────────────────
ONTOLOGY="${1:-mp}"
ROOT_TERM="${2:-mammalian phenotype}"
# ──────────────────────────────────────────────────────────────────────────────

EXPERT_EXPANDED_DIR="${EVAL_ROOT}/modules/01.eval_experts/outputs"
EXPERT_ORIG_DIR="${EVAL_ROOT}/modules/01.eval_experts/inputs/${ONTOLOGY}"
LLM_EXPANDED_DIR="${MODULE_ROOT}/outputs"
LLM_ORIG_DIR="${MODULE_ROOT}/outputs"

OWL_FILE="${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/owl_files/${ONTOLOGY}.ttl"
ONTOLOGY_EMBEDDING_DB="${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/kalm/${ONTOLOGY}.db"
OUTPUT_DIR="${MODULE_ROOT}/outputs/figures"

module purge 2>/dev/null || true
module load gcc 2>/dev/null || true

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
if [ -d "$GCC_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH
fi

# Activate environment
source "${EVAL_ROOT}/env-evaluation/bin/activate"

echo "=== Evaluating Combined Responses (Humans + LLMs) ==="
echo "  Ontology           : ${ONTOLOGY}"
echo "  OWL file           : ${OWL_FILE}"
echo "  Expert Expanded dir: ${EXPERT_EXPANDED_DIR}"
echo "  LLM Expanded dir   : ${LLM_EXPANDED_DIR}"
echo "  Figures dir        : ${OUTPUT_DIR}"
echo "  Manual terms       : excluded from evaluation/plots"
echo "  Jaccard boxplot    : skipped"
echo "  Pattern summary    : combined A-D panels plus standalone selected-term excess panel"
echo "  Panel C colour     : human-LLM Jaccard similarity"

# Remove stale outputs from previous figure layouts.
rm -f \
    "${OUTPUT_DIR}/term_variability_pattern_summary_abc.png" \
    "${OUTPUT_DIR}/term_variability_pattern_summary_b.png" \
    "${OUTPUT_DIR}/term_variability_pattern_summary_de.png"

python3 "${MODULE_ROOT}/scripts/05.eval_combined.py" \
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
    --skip_inter_annotator_agreement_boxplot

echo "Done. Combined plots written to: ${OUTPUT_DIR}"
echo "  - term_variability_pattern_summary.png"
echo "  - term_variability_pattern_summary_c.png"
echo "  - model_comparison_table.png"
echo "  - speedup_comparison_table.png"
