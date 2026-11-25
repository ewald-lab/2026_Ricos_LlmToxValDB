#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --mem=80G
#SBATCH -J "toxval_llm_production_run"
#SBATCH -o "logs/out_tvllm_prod_%j.out"
#SBATCH -e "logs/err_tvllm_prod_%j.err"

# Arguments: endpoint (required), model (optional), port (optional), ontology (optional)
#
# The ontology is any id in toxvaldb/config/ontologies.json (default: mp).
# To map several ontologies, submit the job as an array without the ontology
# argument: each task takes the registry entry at its index, e.g.
#   sbatch --array=0-13 00.hipo_toxvaldb.sh <host> [model] [port]
# runs the 14 ontologies the registry has today. The job log prints the index
# map, and an index past the end of the registry stops the task.
ENDPOINT="${1:-}"
MODEL="${2:-Qwen3.6-35B-A3B-FP8}"
PORT="${3:-8053}"
ONTOLOGY_ARG="${4:-}"

if [ -z "$ENDPOINT" ]; then
    echo "Error: Host node / endpoint is required."
    echo "Usage: sbatch $0 <host|base_url> [model] [port] [ontology]"
    echo "Example: sbatch $0 http://localhost:8000/v1 Qwen3.6-35B-A3B-FP8 8000 mp"
    echo "All ontologies: sbatch --array=0-13 $0 <host|base_url> [model] [port]"
    exit 1
fi

CURRENT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$CURRENT_DIR"
while [ "$REPO_ROOT" != "/" ] && [ ! -d "$REPO_ROOT/HiPO" ]; do
    REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
done

if [ ! -d "$REPO_ROOT/HiPO" ]; then
    echo "Error: Could not locate repository root containing HiPO directory."
    exit 1
fi

export TOXVALDB_ROOT="${REPO_ROOT}/toxvaldb"
HIPO_DIR="${REPO_ROOT}/HiPO"
MODULE_ROOT="${TOXVALDB_ROOT}/01.hipo/modules/00.hipo_toxvaldb"

source "${TOXVALDB_ROOT}/cluster_env.sh"
BASE_URL="$(resolve_base_url "${ENDPOINT}" "${PORT}")"

# The ontology ids and the location of their databases come from
# toxvaldb/config/ontologies.json.
source "${TOXVALDB_ROOT}/config/ontologies.sh" || exit 1
PREP_ONTO_ROOT="${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies"

# Only the ontology id and its database folder are needed here. The ontology
# source file is not read, so it is not resolved and no snapshot is required.
if [ -n "${ONTOLOGY_ARG}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: give an ontology or submit an array, not both." >&2
    echo "  Every array task would map '${ONTOLOGY_ARG}' and write the same output files." >&2
    exit 1
fi
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ONTOLOGY_LIST="$(ontology_group_ids all)" || exit 1
    mapfile -t ONTOLOGY_CHOICES <<< "${ONTOLOGY_LIST}"
    ONTOLOGY="${ONTOLOGY_CHOICES[${SLURM_ARRAY_TASK_ID}]:-}"
    ontology_echo_index_map all
    if [ -z "${ONTOLOGY}" ]; then
        echo "Error: array index ${SLURM_ARRAY_TASK_ID} is outside 0-$(( ${#ONTOLOGY_CHOICES[@]} - 1 ))." >&2
        exit 1
    fi
else
    ONTOLOGY="${ONTOLOGY_ARG:-mp}"
fi
ONTOLOGY_DB_ROOT="$(ontology_db_root_for "${ONTOLOGY}" "${PREP_ONTO_ROOT}")" || exit 1

echo "--- Processing Ontology: ${ONTOLOGY} (Task ID: ${SLURM_ARRAY_TASK_ID:-single}) ---"
echo "  Ontology databases: ${ONTOLOGY_DB_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

setup_cluster_env cuda/12.8.0 gcc

VENV_DIR="${TOXVALDB_ROOT}/01.hipo/env-text-to-onto"
source "${VENV_DIR}/bin/activate"

mkdir -p "${MODULE_ROOT}/jobs/logs"
mkdir -p "${MODULE_ROOT}/outputs"

"${VENV_DIR}/bin/python" "${HIPO_DIR}/hipo.py" \
    --input_file "${TOXVALDB_ROOT}/00.data_preparation/modules/01.preprocess_toxvaldb/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv" \
    --output_file "${MODULE_ROOT}/outputs/ToxValDB_Classified_${ONTOLOGY}_${MODEL}.csv" \
    --reasoning_file "${MODULE_ROOT}/outputs/ToxValDB_Reasoning_${ONTOLOGY}_${MODEL}.json" \
    --target_column TOXICOLOGICAL_EFFECT_LLM_CLEANED \
    --system_prompt_file "${MODULE_ROOT}/config/system_prompt_{MODE}.txt" \
    --rag_config "${MODULE_ROOT}/inputs/choice_rag.json" \
    --ontology_embedding_db "${ONTOLOGY_DB_ROOT}/{EMBED_DIR}/${ONTOLOGY}.db" \
    --global_definitions_db "${ONTOLOGY_DB_ROOT}/definitions_${ONTOLOGY}.db" \
    --target_embedding_db "${TOXVALDB_ROOT}/00.data_preparation/modules/01.preprocess_toxvaldb/outputs/embeddings_db/tox_effects_{EMBED_DIR}.db" \
    --base_url "${BASE_URL}" \
    --num_backups 10 \
    --context_ontology_file "${PREP_ONTO_ROOT}/inputs/ontology_context/${ONTOLOGY}.txt" \
    --model "${MODEL}" \
    --llm_config "${MODULE_ROOT}/inputs/choice_llm.json" \
    --descriptions_concurrency 100 \
    --ontology_definitions
