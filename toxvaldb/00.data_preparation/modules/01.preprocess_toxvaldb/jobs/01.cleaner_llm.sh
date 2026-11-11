#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH -J "clean_llm"
#SBATCH -o "logs/clean_llm%j.out"
#SBATCH -e "logs/clean_llm%j.err"

if [ -n "$SLURM_JOB_ID" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TOXVALDB_ROOT="$(cd "${PREP_ROOT}/.." && pwd)"
PROJECT_ROOT="$(cd "${TOXVALDB_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

source "${TOXVALDB_ROOT}/cluster_env.sh"

mkdir -p "${SCRIPT_DIR}/logs"

ENDPOINT="${1:-${CODON_NODE:-${BASE_URL:-}}}"
MODEL_CONFIG="${2:-${MODEL_CONFIG:-}}"
PORT="${3:-${CODON_PORT:-8003}}"

if [ -z "$ENDPOINT" ] || [ -z "$MODEL_CONFIG" ]; then
    echo "Error: Both host/base_url and model_config are required."
    echo "Usage: sbatch $0 <host|base_url> <model_config> [port]"
    echo "Example: sbatch $0 http://localhost:8000/v1 HiPO/config/models/Qwen3.6-35B-A3B-FP8.json"
    exit 1
fi

FINAL_BASE_URL="$(resolve_base_url "${ENDPOINT}" "${PORT}")"

if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/env-data-prep/bin/activate"
elif [ -f "${PREP_ROOT}/../env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/../env-data-prep/bin/activate"
fi

echo "=== LLM Cleaning (Positives) ==="
echo "Endpoint URL : ${FINAL_BASE_URL}"
echo "Model Config : ${MODEL_CONFIG}"

python "${REPO_ROOT}/scripts/01.llm_clean.py" \
    --input "${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_01.tsv" \
    --output "${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_02_positives.tsv" \
    --reasoning_file "${REPO_ROOT}/outputs/llm_clean_reasoning_positives.json" \
    --model_config "${MODEL_CONFIG}" \
    --pipeline_dir "${REPO_ROOT}/config/clean_prompts/pipeline" \
    --system_prompt_exceptions_file "${REPO_ROOT}/config/clean_prompts/system_prompt_clean_exceptions.txt" \
    --base_url "${FINAL_BASE_URL}"
