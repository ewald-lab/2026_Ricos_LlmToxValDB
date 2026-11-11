#!/bin/bash
set -euo pipefail

# Portable vLLM environment setup (Linux + NVIDIA CUDA GPU required to serve models)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-${SCRIPT_DIR}/env-vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: 'uv' is required but was not found on PATH."
    echo "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Creating vLLM virtual environment at: ${ENV_DIR}"
uv venv "${ENV_DIR}" --python="${PYTHON_VERSION}"

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

echo "Installing vLLM and Hugging Face Hub CLI..."
uv pip install --no-cache -U vllm huggingface_hub

echo ""
echo "vLLM environment ready at: ${ENV_DIR}"
echo "Activate with: source ${ENV_DIR}/bin/activate"

