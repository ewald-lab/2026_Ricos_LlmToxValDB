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

SCRIPT_DIR=$(dirname "$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)")
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge && module load cuda/12.8.0 gcc

source "${PREP_ROOT}/env-data-prep/bin/activate"

EMBED_MODEL="${MODELS_KEYS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DB="${REPO_ROOT}/outputs/embeddings_db/tox_effects_${EMBED_MODEL}.db"

echo "=========================================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID | Model: $EMBED_MODEL"
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
