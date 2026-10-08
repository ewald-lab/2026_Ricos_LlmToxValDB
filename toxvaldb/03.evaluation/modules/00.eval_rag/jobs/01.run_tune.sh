#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -J "rag_tune_fast"
#SBATCH -o "logs/tune_rag_fast_%A_%a.out"
#SBATCH -e "logs/tune_rag_fast_%A_%a.err"
#SBATCH --array=0-47

if [[ "${BASH_SOURCE[0]}" == /var/spool/slurmd/* ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"
TOXVALDB_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${TOXVALDB_ROOT}/.." && pwd)"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-/nfs/research/ewald/ricos/huggingface_cache}"

module purge && module load cuda/12.8.0 gcc

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
if [ -d "$GCC_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH
fi

source "${EVAL_ROOT}/env-evaluation/bin/activate"

mkdir -p "${SCRIPT_DIR}/logs" "${MODULE_ROOT}/outputs/cache" "${MODULE_ROOT}/outputs/results_tune"

# Evaluation Configuration & Data paths
GROUND_TRUTH="${MODULE_ROOT}/inputs/ground_truth_mp_expanded.csv"
EVAL_CONFIG="${MODULE_ROOT}/config/eval_rag.json"
RERANKER_CONFIG="${MODULE_ROOT}/config/reranker_prompt.txt"
DEFINITIONS_DB="${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/definitions_mp.db"
TOXVAL_DB="${TOXVALDB_ROOT}/00.data_preparation/modules/01.preprocess_toxvaldb/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv"

# Determine embedding model, reranker model, and data variant for current task
# Layout: 4 models × 3 rerankers × 4 variants = 48 tasks
MODELS=("microsoft/harrier-oss-v1-27b" "tencent/KaLM-Embedding-Gemma3-12B-2511" "nvidia/llama-embed-nemotron-8b" "Qwen/Qwen3-Embedding-8B")
RERANKERS=("Qwen/Qwen3-Reranker-0.6B" "Qwen/Qwen3-Reranker-4B" "Qwen/Qwen3-Reranker-8B")
DATA_VARIANTS=("base" "def" "hier" "def_hier")

NUM_RERANKERS=${#RERANKERS[@]}
NUM_VARIANTS=${#DATA_VARIANTS[@]}

TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:-0}}"

EMB_IDX=$((TASK_ID / (NUM_RERANKERS * NUM_VARIANTS)))
REMAINDER=$((TASK_ID % (NUM_RERANKERS * NUM_VARIANTS)))
RERANK_IDX=$((REMAINDER / NUM_VARIANTS))
VARIANT_IDX=$((REMAINDER % NUM_VARIANTS))

MODEL=${MODELS[$EMB_IDX]}
RERANKER_MODEL=${RERANKERS[$RERANK_IDX]}
DATA_VARIANT=${DATA_VARIANTS[$VARIANT_IDX]}

case $MODEL in
    "microsoft/harrier-oss-v1-27b") DIR="harrier" ;;
    "tencent/KaLM-Embedding-Gemma3-12B-2511") DIR="kalm" ;;
    "nvidia/llama-embed-nemotron-8b") DIR="nemotron" ;;
    "Qwen/Qwen3-Embedding-8B") DIR="qwen" ;;
    *) DIR=$(echo "$MODEL" | tr / _) ;;
esac

case $RERANKER_MODEL in
    "Qwen/Qwen3-Reranker-0.6B") RERANK_SHORT="qwen3_0.6b" ;;
    "Qwen/Qwen3-Reranker-4B")   RERANK_SHORT="qwen3_4b" ;;
    "Qwen/Qwen3-Reranker-8B")   RERANK_SHORT="qwen3_8b" ;;
    *) RERANK_SHORT=$(echo "$RERANKER_MODEL" | tr / _) ;;
esac

ONTOLOGY_DB="${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/${DIR}/mp.db"
TARGET_DB="${TOXVALDB_ROOT}/00.data_preparation/modules/01.preprocess_toxvaldb/outputs/embeddings_db/tox_effects_${DIR}.db"
CACHE_FILE="${MODULE_ROOT}/outputs/cache/precomputed_scores_${DIR}_${RERANK_SHORT}_${DATA_VARIANT}.json"
RESULTS_FILE="${MODULE_ROOT}/outputs/results_tune/tuning_results_${DIR}_${RERANK_SHORT}_${DATA_VARIANT}.json"

echo "=== RAG Tuning Task ID: ${TASK_ID} ==="
echo "  Embedding Model : ${MODEL} (${DIR})"
echo "  Reranker Model  : ${RERANKER_MODEL} (${RERANK_SHORT})"
echo "  Data Variant    : ${DATA_VARIANT}"
echo "  Ontology DB     : ${ONTOLOGY_DB}"
echo "  Target DB       : ${TARGET_DB}"
echo "  Cache File      : ${CACHE_FILE}"
echo "  Results File    : ${RESULTS_FILE}"

# Step 1: Precompute Top 150 candidates and Reranker scores
echo "Starting precomputation..."
python3 "${MODULE_ROOT}/scripts/01.precompute_eval_data.py" \
    --ground_truth "$GROUND_TRUTH" \
    --ontology_db "$ONTOLOGY_DB" \
    --target_embedding_db "$TARGET_DB" \
    --embedding_model "$MODEL" \
    --reranker_model "$RERANKER_MODEL" \
    --reranker_config "$RERANKER_CONFIG" \
    --toxval_db "$TOXVAL_DB" \
    --global_definitions_db "$DEFINITIONS_DB" \
    --data_variant "$DATA_VARIANT" \
    --output "$CACHE_FILE"

# Step 2: Run Fast Grid Search
echo "Starting fast grid search for all configurations..."
python3 "${MODULE_ROOT}/scripts/02.fast_grid_search.py" \
    --precomputed_data "$CACHE_FILE" \
    --config_grid "$EVAL_CONFIG" \
    --ontology_db "$ONTOLOGY_DB" \
    --data_variant "$DATA_VARIANT" \
    --output "$RESULTS_FILE"

echo "Tuning finished for $MODEL with $RERANKER_MODEL (variant: $DATA_VARIANT). Results saved to: $RESULTS_FILE"
