#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH -J "clean_llm"
#SBATCH -o "logs/clean_llm%j.out"
#SBATCH -e "logs/clean_llm%j.err"

SCRIPT_DIR=$(dirname "$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)")
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CODON_NODE="${1:-${CODON_NODE:-}}"
if [ -z "$CODON_NODE" ]; then
    echo "Error: CODON_NODE is required. Pass it as the first argument or set the CODON_NODE env variable."
    echo "Usage: sbatch $0 <codon-node>  (e.g. sbatch $0 hl-codon-gpu-020)"
    exit 1
fi
BASE_URL="http://${CODON_NODE}.ebi.ac.uk:8003/v1"

source "${PREP_ROOT}/env-data-prep/bin/activate"

echo "=== LLM Cleaning ==="
python "${REPO_ROOT}/scripts/01.llm_clean.py" \
    --input "${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_01.tsv" \
    --output "${REPO_ROOT}/outputs/ToxValDB_v9.7.0_Cleaned_02_positives.tsv" \
    --reasoning_file "${REPO_ROOT}/outputs/llm_clean_reasoning_positives.json" \
    --model_config "/nfs/research/ewald/ricos/code/playground/models/config/Qwen3.5-122B-A10B-FP8.json" \
    --pipeline_dir "${REPO_ROOT}/config/clean_prompts/pipeline" \
    --system_prompt_standardize_file "${REPO_ROOT}/config/clean_prompts/system_prompt_standardize.txt" \
    --system_prompt_exceptions_file "${REPO_ROOT}/config/clean_prompts/system_prompt_clean_exceptions.txt" \
    --base_url "${BASE_URL}"
