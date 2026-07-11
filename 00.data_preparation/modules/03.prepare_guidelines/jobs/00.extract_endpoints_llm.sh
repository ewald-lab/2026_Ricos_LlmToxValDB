#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH -J "extract_endpoints_llm"
#SBATCH -o "logs/out_extract_endpoints_%j.out"
#SBATCH -e "logs/err_extract_endpoints_%j.err"

SCRIPT_DIR=$(dirname "$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)")
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CODON_NODE="${1:-${CODON_NODE:-hl-codon-gpu-020}}"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge && module load gcc

source "${PREP_ROOT}/env-data-prep/bin/activate"

mkdir -p "${SCRIPT_DIR}/logs"

python "${REPO_ROOT}/scripts/00.extract_endpoints_llm.py" \
    --model "Qwen3.6-35B-A3B" \
    --base_url "http://${CODON_NODE}.ebi.ac.uk:8012/v1" \
    --model_config "/nfs/research/ewald/ricos/code/playground/models/config/Qwen3.6-35B-A3B.json" \
    --skill_file "${REPO_ROOT}/inputs/SKILL.md" \
    --inputs_dir "${REPO_ROOT}/inputs/OECD Guidelines" \
    --outputs_dir "${REPO_ROOT}/outputs"
