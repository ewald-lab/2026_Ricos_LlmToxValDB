#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "eval_llms"
#SBATCH --array=0-9
#SBATCH -o "logs/eval_llms_%A_%a.out"
#SBATCH -e "logs/eval_llms_%A_%a.err"

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

source "${TOXVALDB_ROOT}/cluster_env.sh"

export PYTHONPATH="${REPO_ROOT}:${SCRIPT_DIR}/../scripts:${PYTHONPATH}"

mkdir -p "${SCRIPT_DIR}/logs"

# ==============================================================================
# Model & Endpoint Configuration
# ==============================================================================
# Get the host or full base_url from the first argument or environment variable, defaulting to localhost
ENDPOINT="${1:-${CODON_NODE:-localhost}}"

# Handle manual runs where SLURM_ARRAY_TASK_ID is not set
INDEX="${SLURM_ARRAY_TASK_ID:-${2:-8}}"

# Define model mapping based on array index (0-9)
case $INDEX in
    0)
        MODEL="GLM-5.1"
        PORT="8000"
        ;;
    1)
        MODEL="gpt-oss-20b"
        PORT="8001"
        ;;
    2)
        MODEL="NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
        PORT="8002"
        ;;
    3)
        MODEL="Qwen3.5-122B-A10B-FP8"
        PORT="8003"
        ;;
    4)
        MODEL="gpt-oss-120b"
        PORT="8004"
        ;;
    5)
        MODEL="Qwen3.5-35B-A3B-FP8"
        PORT="8005"
        ;;
    6)
        MODEL="Qwen3.6-35B-A3B-FP8"
        PORT="8006"
        ;;
    7)
        MODEL="NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
        PORT="8007"
        ;;
    8)
        MODEL="Qwen3.5-27B-FP8"
        PORT="8008"
        ;;
    9)
        MODEL="Qwen3.6-27B-FP8"
        PORT="8009"
        ;;
    *)
        echo "Error: Invalid index/SLURM_ARRAY_TASK_ID: $INDEX"
        exit 1
        ;;
esac

BASE_URL="${BASE_URL:-$(resolve_base_url "${ENDPOINT}" "${PORT}")}"

# ==============================================================================
# Fixed Parameters
# ==============================================================================
ONTOLOGY="mp"
EMBED_DIR="kalm"
RAG_CONFIG="${MODULE_ROOT}/inputs/best_config.json"
ONTOLOGY_DB="${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/${EMBED_DIR}/${ONTOLOGY}.db"
CONTEXT_FILE="${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt"
GLOBAL_DEFINITIONS_DB="${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/definitions_${ONTOLOGY}.db"

# Other resolved fixed paths
INPUT_FILE="${EVAL_ROOT}/modules/01.eval_experts/outputs/Expanded_average_human.csv"
MODEL_CONFIG="${REPO_ROOT}/HiPO/config/models/${MODEL}.json"
TARGET_EMBEDDING_DB="${MODULE_ROOT}/outputs/tox_effects_embeddings_${EMBED_DIR}.db"
CANDIDATE_POOL_FILE="${TOXVALDB_ROOT}/02.expert_annotation/modules/01.create_form/outputs/expert_annotation_${ONTOLOGY}.html"

# Environment & hardware variables
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# Load system environments
setup_cluster_env cuda/12.8.0 gcc

# Activate evaluation environment
source "${EVAL_ROOT}/env-evaluation/bin/activate"

cd "${MODULE_ROOT}"

# ==============================================================================
# Loop Parameters & Execution
# ==============================================================================
for CONSENSUS in 1 3 5 10; do
    for MODE in 1 2; do
        for DEFS in true false; do
            # Configure ontology definitions argument and filename suffix
            if [ "$DEFS" = "true" ]; then
                DEFS_ARG="--ontology_definitions"
                DEFS_SUFFIX="with_defs"
            else
                DEFS_ARG=""
                DEFS_SUFFIX="no_defs"
            fi

            # Define system prompt based on mode
            SYSTEM_PROMPT="${MODULE_ROOT}/config/system_prompt_${MODE}.txt"

            # Outputs matching expert evaluation formats, parameterized by loop variables
            mkdir -p "${MODULE_ROOT}/outputs/${MODEL}"
            OUTPUT_FILE="${MODULE_ROOT}/outputs/${MODEL}/ToxValLlm_llm_${MODEL}_ontology_${ONTOLOGY}_mode_${MODE}_consensus_${CONSENSUS}_${DEFS_SUFFIX}.csv"
            REASONING_FILE="${MODULE_ROOT}/outputs/${MODEL}/ToxValLlm_llm_${MODEL}_ontology_${ONTOLOGY}_mode_${MODE}_consensus_${CONSENSUS}_${DEFS_SUFFIX}_reasoning.json"

            echo "================================================================================"
            echo "Running evaluation (NEW prompts):"
            echo "  Model             : ${MODEL}"
            echo "  Ontology          : ${ONTOLOGY}"
            echo "  Mode              : ${MODE}"
            echo "  Consensus Runs    : ${CONSENSUS}"
            echo "  Definitions       : ${DEFS}"
            echo "  System Prompt     : ${SYSTEM_PROMPT}"
            echo "  Output File       : ${OUTPUT_FILE}"
            echo "  Manual Terms      : none; LLM outputs only Selected_Candidates"
            echo "================================================================================"

            python3 "${MODULE_ROOT}/scripts/01.eval_llms.py" \
                --input_file "${INPUT_FILE}" \
                --output_file "${OUTPUT_FILE}" \
                --reasoning_file "${REASONING_FILE}" \
                --target_column "Phrase" \
                --model_config "${MODEL_CONFIG}" \
                --base_url "${BASE_URL}" \
                --system_prompt_file "${SYSTEM_PROMPT}" \
                --mode ${MODE} \
                --consensus_runs ${CONSENSUS} \
                ${DEFS_ARG} \
                --context_ontology "$(cat ${CONTEXT_FILE})" \
                --rag_config "${RAG_CONFIG}" \
                --embeddings_model_id "${EMBED_DIR}" \
                --ontology_embedding_db "${ONTOLOGY_DB}" \
                --target_embedding_db "${TARGET_EMBEDDING_DB}" \
                --global_definitions_db "${GLOBAL_DEFINITIONS_DB}" \
                --candidate_pool_file "${CANDIDATE_POOL_FILE}" \
                --descriptions_concurrency 10
        done
    done
done
