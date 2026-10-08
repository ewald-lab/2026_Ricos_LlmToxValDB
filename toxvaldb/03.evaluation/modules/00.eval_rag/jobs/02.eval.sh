#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH -J "rag_eval_summary"
#SBATCH -o "logs/eval_summary_%j.out"
#SBATCH -e "logs/eval_summary_%j.err"

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

mkdir -p "${SCRIPT_DIR}/logs" "${MODULE_ROOT}/outputs/best_configs" "${MODULE_ROOT}/outputs/tables" "${MODULE_ROOT}/outputs/figures/decisions"

# Execute from scripts directory since evaluation scripts output relative to scripts/
cd "${MODULE_ROOT}/scripts"

echo "=== Step 1: Summarizing best configurations across all models ==="
python3 03.summarize_best_configs.py

echo "=== Step 2: Generating model comparison highlighted plot ==="
python3 04.plot_model_comparison_highlighted.py

echo "=== Step 3: Generating decision impact plots ==="
python3 06.plot_decision_impact.py

echo "=== Evaluation summary complete ==="
