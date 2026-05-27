#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH -J "clean_llm"
#SBATCH -o "logs/clean_llm%j.out"
#SBATCH -e "logs/clean_llm%j.err"

# --- Configuration ---
SCRIPT_PATH=$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PREPROCESSED_FILE="${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_01.tsv"
OUTPUT_FILE="${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_02_positives.tsv"
REASONING_FILE="${REPO_ROOT}/outputs/llm_clean_reasoning_positives.json"
MODEL_CONFIG="/nfs/research/ewald/ricos/code/playground/models/config/Qwen3.5-122B-A10B-FP8.json"
PIPELINE_DIR="${REPO_ROOT}/config/clean_prompts/pipeline"
SYSTEM_PROMPT_STANDARIZE="${REPO_ROOT}/config/clean_prompts/system_prompt_standardize.txt"
SYSTEM_PROMPT_EXCEPTIONS="${REPO_ROOT}/config/clean_prompts/system_prompt_clean_exceptions.txt"
CODON_NODE="${1:-${CODON_NODE:-}}"
if [ -z "$CODON_NODE" ]; then
    echo "Error: CODON_NODE is required. Pass it as the first argument or set the CODON_NODE env variable."
    echo "Usage: sbatch $0 <codon-node>  (e.g. sbatch $0 hl-codon-gpu-020)"
    exit 1
fi
BASE_URL="http://${CODON_NODE}.ebi.ac.uk:8003/v1"
DEBUG_MODE="" # --debug Change to "" for full dataset run

# --- Environment Setup ---
source "${PREP_ROOT}/env-data-prep/bin/activate"

export PYTHONPATH="${REPO_ROOT}/scripts:$PYTHONPATH"

echo "=== LLM Cleaning ==="
python "${REPO_ROOT}/scripts/01.llm_clean.py" \
    --input "${PREPROCESSED_FILE}" \
    --output "${OUTPUT_FILE}" \
    --reasoning_file "${REASONING_FILE}" \
    --model_config "${MODEL_CONFIG}" \
    --pipeline_dir "${PIPELINE_DIR}" \
    --system_prompt_standardize_file "${SYSTEM_PROMPT_STANDARIZE}" \
    --system_prompt_exceptions_file "${SYSTEM_PROMPT_EXCEPTIONS}" \
    --base_url "${BASE_URL}" \
    ${DEBUG_MODE}

if [ $? -ne 0 ]; then
    echo "Error: LLM cleaning failed."
    exit 1
fi
echo "LLM cleaning completed: ${OUTPUT_FILE}"
echo "Reasoning saved to: ${REASONING_FILE}"
