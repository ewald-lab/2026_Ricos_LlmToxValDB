#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH -J "toxval_embed_prep"
#SBATCH -o "logs/out_embed_%A_%a.out"
#SBATCH -e "logs/err_embed_%A_%a.err"
#SBATCH --array=0-3

MODELS_KEYS=("qwen" "nemotron" "harrier" "kalm")

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

# Optional arguments: $1 can be model override or custom HF_HOME
if [ -n "$1" ] && [ -d "$1" ]; then
    export HF_HOME="$1"
    MODEL_OVERRIDE="${2:-}"
else
    MODEL_OVERRIDE="${1:-}"
fi

module purge && module load cuda/12.8.0 gcc 2>/dev/null || true

if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/env-data-prep/bin/activate"
elif [ -f "${PREP_ROOT}/../env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/../env-data-prep/bin/activate"
fi

if [ -n "$MODEL_OVERRIDE" ]; then
    EMBED_MODEL="$MODEL_OVERRIDE"
else
    TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    EMBED_MODEL="${MODELS_KEYS[$TASK_ID]}"
fi

OUTPUT_DB="${REPO_ROOT}/outputs/embeddings_db/tox_effects_${EMBED_MODEL}.db"

echo "=========================================================="
echo "Task ID: ${SLURM_ARRAY_TASK_ID:-manual} | Model: $EMBED_MODEL"
echo "HF_HOME: $HF_HOME"
echo "=========================================================="

mkdir -p "$(dirname "${OUTPUT_DB}")"

python "${REPO_ROOT}/scripts/03.embeddings.py" \
    --input_file "${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv" \
    --target_column "TOXICOLOGICAL_EFFECT_LLM_CLEANED" \
    --std_column "TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED" \
    --rag_config "${REPO_ROOT}/config/rag/embeddings.json" \
    --embeddings_model_id "${EMBED_MODEL}" \
    --target_embedding_db "${OUTPUT_DB}" \
    --hf_home "${HF_HOME}"
