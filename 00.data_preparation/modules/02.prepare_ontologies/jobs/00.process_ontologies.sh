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
#SBATCH --array=0-2

MODELS=("kalm" "nemotron" "harrier")

if [ -n "$SLURM_JOB_ID" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${SCRIPT_DIR}/logs"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

module purge 2>/dev/null || true
module load cuda/12.8.0 gcc 2>/dev/null || true

if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/env-data-prep/bin/activate"
fi

# Optional argument overrides: $1 = ontology (default: mp), $2 = embedding model
ONTOLOGY="${1:-mp}"
if [ -n "$2" ]; then
    EMBED_DIR="$2"
else
    TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    EMBED_DIR="${MODELS[$TASK_ID]}"
fi

mkdir -p "${REPO_ROOT}/outputs/${EMBED_DIR}"

echo "=========================================================="
echo "Ontology: ${ONTOLOGY} | Model: ${EMBED_DIR} | Array ID: ${SLURM_ARRAY_TASK_ID:-manual}"
echo "HF_HOME: ${HF_HOME}"
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
    --hf_home "${HF_HOME}"
