#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH -J "extract_endpoints_llm"
#SBATCH -o "logs/out_extract_endpoints_%j.out"
#SBATCH -e "logs/err_extract_endpoints_%j.err"

JOBS_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="$(cd "${JOBS_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge
module load gcc

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH

source "${PREP_ROOT}/env-data-prep/bin/activate"

export PYTHONPATH="${REPO_ROOT}/scripts:$PYTHONPATH"

mkdir -p "${REPO_ROOT}/outputs/endpoints"
mkdir -p "${JOBS_DIR}/logs"

MODEL="Qwen3.5-122B-A10B-FP8"
BASE_URL="http://codon-gpu-001.ebi.ac.uk:8003/v1"
MODEL_CONFIG="${REPO_ROOT}/config/models/${MODEL}.json"

python "${REPO_ROOT}/scripts/00.extract_endpoints_llm.py" \
    --model "${MODEL}" \
    --base_url "${BASE_URL}" \
    --model_config "${MODEL_CONFIG}" \
    --skill_file "${REPO_ROOT}/inputs/SKILL.md" \
    --inputs_dir "${REPO_ROOT}/inputs/OECD Guidelines" \
    --outputs_dir "${REPO_ROOT}/outputs/endpoints"
