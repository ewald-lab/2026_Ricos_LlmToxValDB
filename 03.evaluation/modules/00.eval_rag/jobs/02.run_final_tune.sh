#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=25G
#SBATCH -J "rag_final_tune"
#SBATCH -o "logs/final_tune_rag_%A_%a.out"
#SBATCH -e "logs/final_tune_rag_%A_%a.err"
#SBATCH --array=0-124

# Environmental setup
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

# Activate environment
cd ../../../
source env-evaluation/bin/activate

# Go to scripts dir
cd modules/00.eval_rag/scripts

# Models to evaluate
MODELS=("microsoft/harrier-oss-v1-27b" "tencent/KaLM-Embedding-Gemma3-12B-2511" "nvidia/llama-embed-nemotron-8b" "Qwen/Qwen3-Embedding-8B")
MODEL_DIRS=("harrier" "kalm" "nemotron" "qwen")

# Evaluation Configuration
GROUND_TRUTH="../inputs/ground_truth_mp_jess_expanded.csv"
EVAL_CONFIG="../config/eval_final_tune.json"
RESULTS_OUT="../outputs/results_final_tune/tuning_results_${SLURM_ARRAY_TASK_ID}.json"
RERANKER_MODEL="Qwen/Qwen3-Reranker-8B"
RERANKER_CONFIG="../config/reranker_prompt.txt" 

# Determine model and DB path for current task
MODEL=$(python3 -c "import json; print(json.load(open('$EVAL_CONFIG'))[$SLURM_ARRAY_TASK_ID]['embedding_model'])")

case $MODEL in
    "microsoft/harrier-oss-v1-27b") DIR="harrier" ;;
    "tencent/KaLM-Embedding-Gemma3-12B-2511") DIR="kalm" ;;
    "nvidia/llama-embed-nemotron-8b") DIR="nemotron" ;;
    "Qwen/Qwen3-Embedding-8B") DIR="qwen" ;;
    *) DIR=$(echo "$MODEL" | tr '/' '_') ;;
esac

ONTOLOGY_DB="/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/00.data_preparation/modules/02.prepare_ontologies/outputs/${DIR}/mp.db"

python3 01.tune_rag.py \
    --ground_truth "$GROUND_TRUTH" \
    --config "$EVAL_CONFIG" \
    --output "$RESULTS_OUT" \
    --config_index "$SLURM_ARRAY_TASK_ID" \
    --ontology_db "$ONTOLOGY_DB" \
    --reranker_model "$RERANKER_MODEL" \
    --reranker_config "$RERANKER_CONFIG"
