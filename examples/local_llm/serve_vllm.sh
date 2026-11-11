#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-${SCRIPT_DIR}/env-vllm}"

if [ -f "${ENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${ENV_DIR}/bin/activate"
fi

if ! command -v vllm >/dev/null 2>&1; then
    echo "Error: 'vllm' command not found. Run ./install_vllm.sh first or activate your vLLM environment."
    exit 1
fi

MODEL_PATH_OR_ID="${1:-${MODEL_ID:-Qwen/Qwen3.6-35B-A3B-FP8}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH_OR_ID%/}")}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
QUANTIZATION="${QUANTIZATION:-auto}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

ARGS=(
    "${MODEL_PATH_OR_ID}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --dtype "${DTYPE}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --max-model-len "${MAX_MODEL_LEN}"
    --enable-prefix-caching
    --trust-remote-code
)

if [ "${QUANTIZATION}" = "auto" ]; then
    if [[ "${MODEL_PATH_OR_ID}" == *FP8* ]] || [[ "${MODEL_PATH_OR_ID}" == *fp8* ]]; then
        ARGS+=(--quantization fp8)
    fi
elif [ -n "${QUANTIZATION}" ] && [ "${QUANTIZATION}" != "none" ]; then
    ARGS+=(--quantization "${QUANTIZATION}")
fi

if [ -n "${REASONING_PARSER}" ]; then
    ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi

echo "---------------------------------------------------------------"
echo "Starting vLLM OpenAI-compatible server"
echo "  Model      : ${MODEL_PATH_OR_ID}"
echo "  Served name: ${SERVED_MODEL_NAME}"
echo "  Endpoint   : http://${HOST}:${PORT}/v1"
echo "  TP Size    : ${TP_SIZE} | Max Len: ${MAX_MODEL_LEN}"
echo "---------------------------------------------------------------"

exec vllm serve "${ARGS[@]}"

