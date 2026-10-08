#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH -J "expert_html_gen"
#SBATCH -o "logs/out_expert_gen_%j.out"
#SBATCH -e "logs/err_expert_gen_%j.err"

# Arguments: ontology (optional: hpath, mp, or omit to process both)
ONTOLOGY_ARG="${1:-}"

if [[ "${BASH_SOURCE[0]}" == /var/spool/slurmd/* ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERT_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"
TOXVALDB_ROOT="$(cd "${EXPERT_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${TOXVALDB_ROOT}/.." && pwd)"

module purge && module load gcc

source "${EXPERT_ROOT}/env-expert-annot/bin/activate"

mkdir -p "${MODULE_ROOT}/logs" "${MODULE_ROOT}/outputs"

if [ -n "$ONTOLOGY_ARG" ]; then
    ONTOLOGIES=("$ONTOLOGY_ARG")
else
    ONTOLOGIES=("hpath" "mp")
fi

for ONTOLOGY in "${ONTOLOGIES[@]}"; do
    echo "=== Generating Expert Annotation Form for: ${ONTOLOGY} ==="

    CONTEXT_FILE="${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt"
    CONTEXT_ONTOLOGY=""
    if [ -f "$CONTEXT_FILE" ]; then
        CONTEXT_ONTOLOGY="$(cat "$CONTEXT_FILE")"
    fi

    python "${MODULE_ROOT}/scripts/01.expert_test.py" \
        --input_pool "${EXPERT_ROOT}/modules/00.create_test/outputs/test_pool_${ONTOLOGY}.json" \
        --ontology_db "${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/qwen/${ONTOLOGY}.db" \
        --global_definitions_db "${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/definitions_${ONTOLOGY}.db" \
        --output_file "${MODULE_ROOT}/outputs/expert_annotation_${ONTOLOGY}.html" \
        --email "ricos@ebi.ac.uk" \
        --context_ontology "${CONTEXT_ONTOLOGY}" \
        --top_reranker 10 \
        --num_phrases 50 \
        --skip_first_level

done
