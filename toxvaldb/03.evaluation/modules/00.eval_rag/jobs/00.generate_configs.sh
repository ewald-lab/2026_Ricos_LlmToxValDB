#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH -J "generate_configs"
#SBATCH -o "logs/generate_configs_%j.out"
#SBATCH -e "logs/generate_configs_%j.err"

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

module purge && module load gcc

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
if [ -d "$GCC_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH
fi

source "${EVAL_ROOT}/env-evaluation/bin/activate"

mkdir -p "${SCRIPT_DIR}/logs" "${MODULE_ROOT}/config"

echo "=== Generating evaluation configurations ==="
python3 "${MODULE_ROOT}/scripts/00.generate_configs.py" \
    --output "${MODULE_ROOT}/config/eval_rag.json" \
    --models "microsoft/harrier-oss-v1-27b" "tencent/KaLM-Embedding-Gemma3-12B-2511" "nvidia/llama-embed-nemotron-8b" "Qwen/Qwen3-Embedding-8B" \
    --top_k 20 50 100 150 \
    --threshold_rag 0.1 0.2 0.3 0.4 0.5 0.6 \
    --threshold_rerank 0.05 0.1 0.15 0.2 0.3 0.4 \
    --top_k_rerank 5 10 15 20 \
    --grf 0.0 0.25 0.5 0.75 1.0 \
    --reranker_models "Qwen/Qwen3-Reranker-0.6B" "Qwen/Qwen3-Reranker-4B" "Qwen/Qwen3-Reranker-8B" \
    --data_variants base def hier def_hier

echo "Configuration generation complete. File saved to: ${MODULE_ROOT}/config/eval_rag.json"
