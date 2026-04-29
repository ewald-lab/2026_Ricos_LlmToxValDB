#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH -J "endpoint_to_phenotype"
#SBATCH -o "logs/out_endpoint_to_phenotype_%j.out"
#SBATCH -e "logs/err_endpoint_to_phenotype_%j.err"

set -euo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)
    SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEXT_TO_ONTO_ROOT="$(cd "${MODULE_ROOT}/../00.text_to_onto" && pwd)"
TEXT_TO_ONTO_PROJECT="$(cd "${MODULE_ROOT}/../.." && pwd)"
GIT_ROOT="$(cd "${TEXT_TO_ONTO_PROJECT}/.." && pwd)"

ONTOLOGY="${1:-mp}"
EMBED_DIR="${2:-qwen}"
MODEL="${3:-Qwen3.5-122B-A10B-FP8}"
BASE_URL="${BASE_URL:-http://codon-gpu-001.ebi.ac.uk:8003/v1}"
DEBUG_LIMIT="${DEBUG_LIMIT:-}"

ENDPOINTS_FILE="${GIT_ROOT}/00.data_preparation/modules/03.prepare_guidelines/outputs/normalized_endpoints/normalized_endpoint_list.csv"
OUTPUT_DIR="${MODULE_ROOT}/outputs"
MODEL_CONFIG="${TEXT_TO_ONTO_ROOT}/config/models/${MODEL}.json"
SYSTEM_PROMPT="${MODULE_ROOT}/config/system_prompt_endpoint_filter.txt"
CONTEXT_FILE="${GIT_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt"

ONTOLOGY_DB="${GIT_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/${EMBED_DIR}/${ONTOLOGY}.db"
DEFINITIONS_DB="${GIT_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/definitions_${ONTOLOGY}_gpt-oss-120b.db"

if [ ! -f "${ONTOLOGY_DB}" ]; then
    ONTOLOGY_DB="${GIT_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/ontology_dbs/${EMBED_DIR}/${ONTOLOGY}.db"
    DEFINITIONS_DB="${GIT_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/ontology_dbs/definitions_${ONTOLOGY}_gpt-oss-120b.db"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-/nfs/research/ewald/ricos/huggingface_cache}"

module purge
module load cuda/12.8.0
module load gcc

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
export LD_LIBRARY_PATH="${GCC_LIB_PATH}:${LD_LIBRARY_PATH:-}"

source "${TEXT_TO_ONTO_PROJECT}/env-text-to-onto/bin/activate"
export PYTHONPATH="${MODULE_ROOT}/scripts:${TEXT_TO_ONTO_ROOT}/scripts:${PYTHONPATH:-}"

DEBUG_ARGS=()
if [ -n "${DEBUG_LIMIT}" ]; then
    DEBUG_ARGS=(--debug_limit "${DEBUG_LIMIT}")
fi

echo "=== Endpoint To Phenotype Filters ==="
echo "Ontology: ${ONTOLOGY}"
echo "Embedding key: ${EMBED_DIR}"
echo "Endpoint table: ${ENDPOINTS_FILE}"
echo "Ontology DB: ${ONTOLOGY_DB}"
echo "Output directory: ${OUTPUT_DIR}/${ONTOLOGY}"

python "${MODULE_ROOT}/scripts/00.endpoints_to_phenotypes.py" \
    --input_endpoints "${ENDPOINTS_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --ontology "${ONTOLOGY}" \
    --ontology_embedding_db "${ONTOLOGY_DB}" \
    --global_definitions_db "${DEFINITIONS_DB}" \
    --model_config "${MODEL_CONFIG}" \
    --base_url "${BASE_URL}" \
    --system_prompt_file "${SYSTEM_PROMPT}" \
    --context_ontology "${CONTEXT_FILE}" \
    --max_paths_per_endpoint 5 \
    "${DEBUG_ARGS[@]}"

echo "Endpoint-to-phenotype filters completed."
