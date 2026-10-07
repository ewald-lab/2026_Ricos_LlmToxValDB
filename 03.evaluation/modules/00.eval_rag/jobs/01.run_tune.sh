#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --mem=100G
#SBATCH -J "rag_tune_fast"
#SBATCH -o "logs/tune_rag_fast_%A_%a.out"
#SBATCH -e "logs/tune_rag_fast_%A_%a.err"
#SBATCH --array=0-47

# Environmental setup
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

# Activate environment
cd ../../../
source env-evaluation/bin/activate

# Go to scripts dir
cd modules/00.eval_rag/scripts

# Evaluation Configuration
GROUND_TRUTH="../inputs/ground_truth_mp_expanded.csv"
EVAL_CONFIG="../config/eval_rag.json"
RERANKER_CONFIG="../config/reranker_prompt.txt" 
DEFINITIONS_DB="/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/00.data_preparation/toxvaldb/modules/02.prepare_ontologies/outputs/definitions_mp.db"

# Determine embedding model, reranker model, and data variant for current task
# Layout: 4 models × 3 rerankers × 4 variants = 48 tasks
MODELS=("microsoft/harrier-oss-v1-27b" "tencent/KaLM-Embedding-Gemma3-12B-2511" "nvidia/llama-embed-nemotron-8b" "Qwen/Qwen3-Embedding-8B")
RERANKERS=("Qwen/Qwen3-Reranker-0.6B" "Qwen/Qwen3-Reranker-4B" "Qwen/Qwen3-Reranker-8B")
DATA_VARIANTS=("base" "def" "hier" "def_hier")

# Decompose task ID: models (outer) × rerankers (middle) × variants (inner)
NUM_RERANKERS=${#RERANKERS[@]}
NUM_VARIANTS=${#DATA_VARIANTS[@]}

EMB_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_RERANKERS * NUM_VARIANTS)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (NUM_RERANKERS * NUM_VARIANTS)))
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
    *) DIR=$(echo "$MODEL" | tr '/' '_') ;;
esac

case $RERANKER_MODEL in
    "Qwen/Qwen3-Reranker-0.6B") RERANK_SHORT="qwen3_0.6b" ;;
    "Qwen/Qwen3-Reranker-4B")   RERANK_SHORT="qwen3_4b" ;;
    "Qwen/Qwen3-Reranker-8B")   RERANK_SHORT="qwen3_8b" ;;
    *) RERANK_SHORT=$(echo "$RERANKER_MODEL" | tr '/' '_') ;;
esac

ONTOLOGY_DB="/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/00.data_preparation/toxvaldb/modules/02.prepare_ontologies/outputs/${DIR}/mp.db"
TARGET_DB="/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/00.data_preparation/toxvaldb/modules/01.preprocess_toxvaldb/outputs/embeddings_db/tox_effects_${DIR}.db"
CACHE_FILE="../outputs/cache/precomputed_scores_${DIR}_${RERANK_SHORT}_${DATA_VARIANT}.json"

# Step 1: Precompute Top 150 candidates and Reranker scores
# This is the heavy part that uses GPU
echo "Starting precomputation for $MODEL with $RERANKER_MODEL (variant: $DATA_VARIANT)..."
python3 01.precompute_eval_data.py \
    --ground_truth "$GROUND_TRUTH" \
    --ontology_db "$ONTOLOGY_DB" \
    --target_embedding_db "$TARGET_DB" \
    --embedding_model "$MODEL" \
    --reranker_model "$RERANKER_MODEL" \
    --reranker_config "$RERANKER_CONFIG" \
    --toxval_db "/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/00.data_preparation/toxvaldb/modules/01.preprocess_toxvaldb/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv" \
    --global_definitions_db "$DEFINITIONS_DB" \
    --data_variant "$DATA_VARIANT" \
    --output "$CACHE_FILE"

# Step 2: Run Fast Grid Search
# This is nearly instant and doesn't need GPU (but we are already on a GPU node)
echo "Starting fast grid search for all configurations..."
python3 02.fast_grid_search.py \
    --precomputed_data "$CACHE_FILE" \
    --config_grid "$EVAL_CONFIG" \
    --ontology_db "$ONTOLOGY_DB" \
    --data_variant "$DATA_VARIANT" \
    --output "../outputs/results_tune/tuning_results_${DIR}_${RERANK_SHORT}_${DATA_VARIANT}.json"

echo "Tuning finished for $MODEL with $RERANKER_MODEL (variant: $DATA_VARIANT). Results saved in outputs/results_tune/"
