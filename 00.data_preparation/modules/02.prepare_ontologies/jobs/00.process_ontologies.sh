#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G
#SBATCH -J "toxval_llm_flow"
#SBATCH -o "logs/out_hierarchy_%A_%a.out"
#SBATCH -e "logs/err_hierarchy_%A_%a.err"
#SBATCH --array=0-3

MODELS=("kalm" "nemotron" "harrier" "qwen")

if [ -n "$SLURM_JOB_ID" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${SCRIPT_DIR}/logs"

ENDPOINT="${1:-${CODON_NODE:-${BASE_URL:-}}}"
MODEL_CONFIG="${2:-${MODEL_CONFIG:-}}"
PORT="${3:-${CODON_PORT:-8003}}"
ONTOLOGY="${4:-${ONTOLOGY:-mp}}"
MODEL_OVERRIDE="${5:-}"

if [ -z "$ENDPOINT" ] || [ -z "$MODEL_CONFIG" ]; then
    echo "Error: Both host node/base_url and model_config are required."
    echo "Usage: sbatch $0 <codon-node|base_url> <model_config> [port] [ontology] [model_override]"
    echo "Example: sbatch $0 codon-gpu-010 /nfs/research/ewald/ricos/code/playground/models/config/Qwen3.6-35B-A3B-FP8.json 8053 mp"
    exit 1
fi

if [[ "$ENDPOINT" =~ ^https?:// ]]; then
    FINAL_BASE_URL="$ENDPOINT"
else
    FINAL_BASE_URL="http://${ENDPOINT}.ebi.ac.uk:${PORT}/v1"
fi

if [ -n "$MODEL_OVERRIDE" ]; then
    EMBED_DIR="$MODEL_OVERRIDE"
else
    TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    EMBED_DIR="${MODELS[$TASK_ID]}"
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

module purge 2>/dev/null || true
module load cuda/12.8.0 gcc 2>/dev/null || true

if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/env-data-prep/bin/activate"
fi

mkdir -p "${REPO_ROOT}/outputs/${EMBED_DIR}"

echo "=========================================================="
echo "=== Processing Ontology with LLM Definitions & Embeddings ==="
echo "Ontology     : ${ONTOLOGY}"
echo "Embed Model  : ${EMBED_DIR} (Array ID: ${SLURM_ARRAY_TASK_ID:-manual})"
echo "Endpoint URL : ${FINAL_BASE_URL}"
echo "Model Config : ${MODEL_CONFIG}"
echo "HF_HOME      : ${HF_HOME}"
echo "=========================================================="

python "${REPO_ROOT}/scripts/01.process_ontology.py" \
    --ontology_file "${REPO_ROOT}/inputs/owl_files/${ONTOLOGY}.ttl" \
    --root_uris_file "${REPO_ROOT}/inputs/root_uris.json" \
    --ontology_db "${REPO_ROOT}/outputs/${EMBED_DIR}/${ONTOLOGY}.db" \
    --rag_config "${REPO_ROOT}/config/rag/embeddings.json" \
    --embeddings_model_id "${EMBED_DIR}" \
    --global_definitions_db "${REPO_ROOT}/outputs/definitions_${ONTOLOGY}.db" \
    --prompt_template "${REPO_ROOT}/config/tools/create_description_prompt.txt" \
    --evaluate_prompt_template "${REPO_ROOT}/config/tools/evaluate_description_prompt.txt" \
    --definitions_report "${REPO_ROOT}/outputs/${ONTOLOGY}_definitions_report.json" \
    --model_config "${MODEL_CONFIG}" \
    --base_url "${FINAL_BASE_URL}" \
    --hf_home "${HF_HOME}"
