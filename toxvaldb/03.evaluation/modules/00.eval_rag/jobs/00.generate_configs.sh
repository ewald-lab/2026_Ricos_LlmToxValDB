#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH -J "generate_configs"
#SBATCH -o "logs/generate_configs_%j.out"
#SBATCH -e "logs/generate_configs_%j.err"

# Environmental setup
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

# Activate environment
cd ../../../
source env-evaluation/bin/activate

# Go to scripts dir
cd modules/00.eval_rag/scripts

# Run config generation
echo "Generating evaluation configurations..."
python3 00.generate_configs.py \
    --output "../config/eval_rag.json" \
    --models "microsoft/harrier-oss-v1-27b" "tencent/KaLM-Embedding-Gemma3-12B-2511" "nvidia/llama-embed-nemotron-8b" "Qwen/Qwen3-Embedding-8B" \
    --top_k 20 50 100 150 \
    --threshold_rag 0.1 0.2 0.3 0.4 0.5 0.6 \
    --threshold_rerank 0.05 0.1 0.15 0.2 0.3 0.4 \
    --top_k_rerank 5 10 15 20 \
    --grf 0.0 0.25 0.5 0.75 1.0 \
    --reranker_models "Qwen/Qwen3-Reranker-0.6B" "Qwen/Qwen3-Reranker-4B" "Qwen/Qwen3-Reranker-8B" \
    --data_variants base def hier def_hier

echo "Configuration generation complete. File saved to ../config/eval_rag.json"
