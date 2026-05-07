#!/bin/bash

#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -J "gen_rag_configs"
#SBATCH -o "logs/gen_configs_%j.out"
#SBATCH -e "logs/gen_configs_%j.err"

# Activate environment
cd ../../../
source env-evaluation/bin/activate

# Go to scripts dir
cd modules/00.eval_rag/scripts

# Generate configurations with specific ranges
N=125
OUTPUT="../config/eval_final_tune.json"

# Ranges (como estaban)
MODELS="microsoft/harrier-oss-v1-27b tencent/KaLM-Embedding-Gemma3-12B-2511 nvidia/llama-embed-nemotron-8b Qwen/Qwen3-Embedding-8B"
TOP_K="10 20 50 100"
THRESH_RAG="0.2 0.3 0.4 0.5"
THRESH_RERANK="0.1 0.3 0.5 0.8"
GRF="0.4 0.6 0.8"

python3 00.generate_configs.py \
    --n $N \
    --output $OUTPUT \
    --models $MODELS \
    --top_k $TOP_K \
    --threshold_rag $THRESH_RAG \
    --threshold_rerank $THRESH_RERANK \
    --grf $GRF

echo "Configurations generated in $OUTPUT"
