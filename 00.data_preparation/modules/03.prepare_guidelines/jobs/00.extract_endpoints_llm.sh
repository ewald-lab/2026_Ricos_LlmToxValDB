#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH -J "extract_endpoints_llm"
#SBATCH -o "logs/out_extract_endpoints_%j.out"
#SBATCH -e "logs/err_extract_endpoints_%j.err"

if [ -n "$SLURM_JOB_ID" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${SCRIPT_DIR}/logs"

ENDPOINT="${1:-${CODON_NODE:-${BASE_URL:-}}}"
MODEL_CONFIG="${2:-${MODEL_CONFIG:-}}"
PORT="${3:-${CODON_PORT:-8003}}"

if [ -z "$ENDPOINT" ] || [ -z "$MODEL_CONFIG" ]; then
    echo "Error: Both host node/base_url and model_config are required."
    echo "Usage: sbatch $0 <codon-node|base_url> <model_config> [port]"
    echo "Example: sbatch $0 codon-gpu-010 /nfs/research/ewald/ricos/code/playground/models/config/Qwen3.6-35B-A3B-FP8.json 8053"
    exit 1
fi

if [[ "$ENDPOINT" =~ ^https?:// ]]; then
    FINAL_BASE_URL="$ENDPOINT"
else
    FINAL_BASE_URL="http://${ENDPOINT}.ebi.ac.uk:${PORT}/v1"
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

module purge 2>/dev/null || true
module load gcc 2>/dev/null || true

if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/env-data-prep/bin/activate"
elif [ -f "${PREP_ROOT}/../env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/../env-data-prep/bin/activate"
fi
fi

echo "=========================================================="
echo "=== Extract Endpoints from Guidelines with LLM ==="
echo "Endpoint URL : ${FINAL_BASE_URL}"
echo "Model Config : ${MODEL_CONFIG}"
echo "Inputs Dir   : ${REPO_ROOT}/inputs/OECD Guidelines"
echo "Outputs Dir  : ${REPO_ROOT}/outputs"
echo "=========================================================="

python "${REPO_ROOT}/scripts/00.extract_endpoints_llm.py" \
    --model_config "${MODEL_CONFIG}" \
    --base_url "${FINAL_BASE_URL}" \
    --skill_file "${REPO_ROOT}/inputs/SKILL.md" \
    --inputs_dir "${REPO_ROOT}/inputs/OECD Guidelines" \
    --outputs_dir "${REPO_ROOT}/outputs"
