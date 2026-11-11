#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL_CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/HiPO/config/models/Qwen3.6-35B-A3B-FP8.json}"
ONTOLOGY="${ONTOLOGY:-mp}"
ONTOLOGY_DB="${ONTOLOGY_DB:-}"
GLOBAL_DEFINITIONS_DB="${GLOBAL_DEFINITIONS_DB:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
CHECK_ONLY=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --model-config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --ontology)
            ONTOLOGY="$2"
            shift 2
            ;;
        --ontology-db)
            ONTOLOGY_DB="$2"
            shift 2
            ;;
        --definitions-db)
            GLOBAL_DEFINITIONS_DB="$2"
            shift 2
            ;;
        --check-only)
            CHECK_ONLY=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--base-url URL] [--model-config JSON] [--ontology mp] [--ontology-db PATH] [--definitions-db PATH] [--check-only]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

HIPO_ENV="${REPO_ROOT}/toxvaldb/01.hipo/env-text-to-onto/bin/activate"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${HIPO_ENV}" ]; then
    # shellcheck disable=SC1091
    source "${HIPO_ENV}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "1. Checking OpenAI-compatible HTTP endpoint: ${BASE_URL}/models"
if ! curl -sf "${BASE_URL}/models" >/dev/null; then
    echo "Error: Could not reach ${BASE_URL}/models"
    echo "Start your vLLM server first (e.g. ./serve_vllm.sh) or set BASE_URL=..."
    exit 1
fi

echo "2. Verifying HiPO LLM client configuration: ${MODEL_CONFIG}"
"${PYTHON_BIN}" - "${MODEL_CONFIG}" "${BASE_URL}" <<'PY'
import json
import sys
from HiPO.model.llm import get_model, test_model

config_path, base_url = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["base_url"] = base_url
llm = get_model(**cfg)
if not test_model(llm, max_retries=1, delay=1):
    raise SystemExit("LLM connectivity test failed.")
PY

if [ "${CHECK_ONLY}" -eq 1 ]; then
    echo "Connectivity check succeeded."
    exit 0
fi

if [ -z "${ONTOLOGY_DB}" ]; then
    DEFAULT_DB="${REPO_ROOT}/toxvaldb/00.data_preparation/modules/02.prepare_ontologies/outputs/qwen/${ONTOLOGY}.db"
    if [ -f "${DEFAULT_DB}" ]; then
        ONTOLOGY_DB="${DEFAULT_DB}"
    fi
fi

if [ -z "${ONTOLOGY_DB}" ] || [ ! -f "${ONTOLOGY_DB}" ]; then
    echo ""
    echo "Connectivity succeeded, but no ontology embedding DB was provided/found."
    echo "To run the full HiPO example, first build an ontology DB with:"
    echo "  toxvaldb/00.data_preparation/modules/02.prepare_ontologies/scripts/01.process_ontology.py"
    echo "Then re-run with:"
    echo "  $0 --ontology-db /path/to/${ONTOLOGY}.db"
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/HiPO/hipo.py"
    --input_file "${SCRIPT_DIR}/sample_effects.tsv"
    --output_file "${OUTPUT_DIR}/sample_classified_${ONTOLOGY}.csv"
    --reasoning_file "${OUTPUT_DIR}/sample_reasoning_${ONTOLOGY}.json"
    --target_column TOXICOLOGICAL_EFFECT
    --system_prompt_file "${REPO_ROOT}/toxvaldb/01.hipo/modules/00.hipo_toxvaldb/config/system_prompt_{MODE}.txt"
    --context_ontology_file "${REPO_ROOT}/toxvaldb/00.data_preparation/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt"
    --ontology_embedding_db "${ONTOLOGY_DB}"
    --target_embedding_db "${OUTPUT_DIR}/sample_effects_embeddings.db"
    --model_config "${MODEL_CONFIG}"
    --base_url "${BASE_URL}"
    --num_backups 0
)

if [ -n "${GLOBAL_DEFINITIONS_DB}" ] && [ -f "${GLOBAL_DEFINITIONS_DB}" ]; then
    CMD+=(--global_definitions_db "${GLOBAL_DEFINITIONS_DB}" --ontology_definitions)
fi

echo "3. Running HiPO example pipeline..."
exec "${CMD[@]}"

