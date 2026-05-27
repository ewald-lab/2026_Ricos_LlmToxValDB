#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH -J "endpoint_to_phenotype"
#SBATCH --array=0-4
#SBATCH -o "logs/out_endpoint_to_phenotype_%A_%a.out"
#SBATCH -e "logs/err_endpoint_to_phenotype_%A_%a.err"

# --- Configuration ---
JOBS_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
MODULE_ROOT="$(cd "${JOBS_DIR}/.." && pwd)"
TEXT_TO_ONTO_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"
PROJECT_ROOT="$(cd "${TEXT_TO_ONTO_ROOT}/.." && pwd)"
DATA_PREP_ROOT="${PROJECT_ROOT}/00.data_preparation"

echo "Job started at: $(date)"
echo "Module Root: ${MODULE_ROOT}"
echo "Data Prep Root: ${DATA_PREP_ROOT}"

CODON_NODE="${1:-${CODON_NODE:-hl-codon-gpu-020}}"
EMBED_DIR="${2:-kalm}"
MODEL="${3:-Qwen3.5-122B-A10B-FP8}"
MODE="${4:-1}"
CONSENSU_RUNS="${5:-3}"
BASE_URL="${BASE_URL:-http://${CODON_NODE}.ebi.ac.uk:8003/v1}"

ENDPOINTS_FILE="${DATA_PREP_ROOT}/modules/03.prepare_guidelines/outputs/normalized_endpoint_list.csv"
OUTPUT_DIR="${MODULE_ROOT}/outputs"
MODEL_CONFIG="/nfs/research/ewald/ricos/code/playground/models/config/${MODEL}.json"
SYSTEM_PROMPT="${MODULE_ROOT}/config/system_prompt_endpoint_filter.txt"
RAG_CONFIG="${MODULE_ROOT}/config/rag/best_config.json"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-/nfs/research/ewald/ricos/huggingface_cache}"

module purge
module load cuda/12.8.0
module load gcc

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
export LD_LIBRARY_PATH="${GCC_LIB_PATH}:${LD_LIBRARY_PATH:-}"

source "${DATA_PREP_ROOT}/env-data-prep/bin/activate"
export PYTHONPATH="${MODULE_ROOT}/scripts:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

# --- Ontology Setup (SLURM Array) ---
ONTOLOGIES=(mp hpath aop hp txpo)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [ -z "${TASK_ID}" ] || [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${#ONTOLOGIES[@]}" ]; then
    echo "Error: Invalid Task ID '${TASK_ID}'. Must be between 0 and $((${#ONTOLOGIES[@]} - 1))"
    exit 1
fi

ONTOLOGY="${ONTOLOGIES[$TASK_ID]}"
echo "--- Processing Ontology: ${ONTOLOGY} (Task ID: ${TASK_ID}) ---"

CONTEXT_FILE="${DATA_PREP_ROOT}/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt"
ONTOLOGY_DB="${DATA_PREP_ROOT}/modules/02.prepare_ontologies/outputs/${EMBED_DIR}/${ONTOLOGY}.db"
DEFINITIONS_DB="${DATA_PREP_ROOT}/modules/02.prepare_ontologies/outputs/definitions_${ONTOLOGY}.db"

if [ ! -f "${ONTOLOGY_DB}" ]; then
    ONTOLOGY_DB="${DATA_PREP_ROOT}/modules/02.prepare_ontologies/inputs/ontology_dbs/${EMBED_DIR}/${ONTOLOGY}.db"
    DEFINITIONS_DB="${DATA_PREP_ROOT}/modules/02.prepare_ontologies/inputs/ontology_dbs/definitions_${ONTOLOGY}.db"
fi

python "${MODULE_ROOT}/scripts/00.guideline_to_onto.py" \
    --input_endpoints "${ENDPOINTS_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --ontology "${ONTOLOGY}" \
    --ontology_embedding_db "${ONTOLOGY_DB}" \
    --global_definitions_db "${DEFINITIONS_DB}" \
    --model_config "${MODEL_CONFIG}" \
    --base_url "${BASE_URL}" \
    --system_prompt_file "${SYSTEM_PROMPT}" \
    --context_ontology "${CONTEXT_FILE}" \
    --rag_config "${RAG_CONFIG}" \
    --embeddings_model_id "${EMBED_DIR}" \
    --mode "${MODE}" \
    --consensus_runs "${CONSENSU_RUNS}"
