#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH -J "toxval_embed_prep"
#SBATCH -o "logs/out_embed_%A_%a.out"
#SBATCH -e "logs/err_embed_%A_%a.err"
#SBATCH --array=0-3%2

MODELS_KEYS=("qwen" "nemotron" "harrier" "kalm")

SCRIPT_PATH=$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge
module load cuda/12.8.0
module load gcc

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH

# --- Environment Setup ---
source "${PREP_ROOT}/env-data-prep/bin/activate"

# Add scripts directory to PYTHONPATH to allow imports of src and model
export PYTHONPATH="${REPO_ROOT}/scripts:$PYTHONPATH"

TASK_ID=$SLURM_ARRAY_TASK_ID
EMBED_MODEL="${MODELS_KEYS[$TASK_ID]}"

RAG_CONFIG="${REPO_ROOT}/config/rag/embeddings.json"
INPUT_FILE="${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv"
OUTPUT_DB="${REPO_ROOT}/outputs/embeddings_db/tox_effects_${EMBED_MODEL}.db"

echo "=========================================================="
echo "Task ID: $TASK_ID | Model: $EMBED_MODEL"
echo "=========================================================="

mkdir -p "$(dirname "${OUTPUT_DB}")"

python "${REPO_ROOT}/scripts/03.embeddings.py" \
    --input_file "${INPUT_FILE}" \
    --target_column "TOXICOLOGICAL_EFFECT_LLM_CLEANED" \
    --std_column "TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED" \
    --rag_config "${RAG_CONFIG}" \
    --embeddings_model_id "${EMBED_MODEL}" \
    --target_embedding_db "${OUTPUT_DB}" \
    --hf_home "${HF_HOME}"
