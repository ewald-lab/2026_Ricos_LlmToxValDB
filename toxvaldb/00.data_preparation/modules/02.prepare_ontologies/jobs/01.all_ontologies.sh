#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH -J "process_all_ontologies"
#SBATCH -o "logs/out_all_ontologies_%A_%a.out"
#SBATCH -e "logs/err_all_ontologies_%A_%a.err"
#SBATCH --array=0-13

# One array task per ontology in toxvaldb/config/ontologies.json ("order").
# The --array range above must cover that list: Slurm parses the directive
# before the script runs, so it cannot be read from the registry.

if [ -n "$SLURM_JOB_ID" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TOXVALDB_ROOT="$(cd "${PREP_ROOT}/.." && pwd)"
PROJECT_ROOT="$(cd "${TOXVALDB_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

source "${TOXVALDB_ROOT}/cluster_env.sh"
# Ontology source, rdflib format, root class, exclusions and DB location all
# come from toxvaldb/config/ontologies.json.
source "${TOXVALDB_ROOT}/config/ontologies.sh" || exit 1

mkdir -p "${SCRIPT_DIR}/logs"

# Optional argument overrides:
# $1: ontology override (default: the registry entry at index SLURM_ARRAY_TASK_ID)
# $2: embedding model (default: qwen)
# $3: custom HF_HOME directory
#
# Optional environment override:
#   ONTOLOGY_SOURCE : "local" or "url"; forces every ontology to its local
#                     snapshot or to its URL instead of the registry's "pin".
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ONTOLOGY_OVERRIDE="${1:-}"
EMBED_DIR="${2:-qwen}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

if [ -n "$3" ] && [ -d "$3" ]; then
    export HF_HOME="$3"
fi

setup_cluster_env cuda/12.8.0 gcc

if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/env-data-prep/bin/activate"
elif [ -f "${PREP_ROOT}/../env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/../env-data-prep/bin/activate"
fi
fi

if [ -n "${ONTOLOGY_OVERRIDE}" ]; then
    ontology_config "${ONTOLOGY_OVERRIDE}" || exit 1
else
    ontology_by_index "${TASK_ID}" || exit 1
fi

OUTPUT_ROOT="$(ontology_db_root "${REPO_ROOT}")" || exit 1
mkdir -p "${OUTPUT_ROOT}/${EMBED_DIR}"

echo "=========================================================="
echo "=== Processing Ontology Embeddings (All Ontologies) ==="
echo "Ontology     : ${ONTOLOGY_ID} (Array ID: ${SLURM_ARRAY_TASK_ID:-manual})"
echo "Embed Model  : ${EMBED_DIR}"
echo "HF_HOME      : ${HF_HOME}"
echo "Output root  : ${OUTPUT_ROOT}"
ontology_echo_index_map
ontology_echo_config
echo "=========================================================="

python "${REPO_ROOT}/scripts/01.process_ontology.py" \
    ${ONTOLOGY_ROOT_URI_FLAG} \
    ${ONTOLOGY_EXCLUDE_URIS_FLAG} \
    ${ONTOLOGY_FORMAT_FLAG} \
    --ontology_file "${ONTOLOGY_FILE}" \
    --ontology_db "${OUTPUT_ROOT}/${EMBED_DIR}/${ONTOLOGY_ID}.db" \
    --rag_config "${REPO_ROOT}/config/rag/embeddings.json" \
    --embeddings_model_id "${EMBED_DIR}" \
    --global_definitions_db "${OUTPUT_ROOT}/definitions_${ONTOLOGY_ID}.db" \
    --hf_home "${HF_HOME}"
