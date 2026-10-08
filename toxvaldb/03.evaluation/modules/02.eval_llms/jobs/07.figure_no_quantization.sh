#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "fig_no_quant"
#SBATCH -o "logs/fig_no_quant_%j.out"
#SBATCH -e "logs/fig_no_quant_%j.err"

# Resolve paths relative to this script location
if [[ "${BASH_SOURCE[0]}" == /var/spool/slurmd/* ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"
TOXVALDB_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${TOXVALDB_ROOT}/.." && pwd)"

mkdir -p "${SCRIPT_DIR}/logs" "${MODULE_ROOT}/outputs/figures"

# Configuration
ONTOLOGY="${1:-mp}"
EXPERT_EXPANDED_DIR="${EVAL_ROOT}/modules/01.eval_experts/outputs"
EXPERT_ORIG_DIR="${EVAL_ROOT}/modules/01.eval_experts/inputs/${ONTOLOGY}"
LLM_EXPANDED_DIR="${MODULE_ROOT}/outputs"
OUTPUT_DIR="${MODULE_ROOT}/outputs/figures"

module purge 2>/dev/null || true
module load gcc 2>/dev/null || true

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
if [ -d "$GCC_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH
fi

# Activate environment
source "${EVAL_ROOT}/env-evaluation/bin/activate"

echo "=== Generating Quantization Comparison Table ==="
echo "  Ontology           : ${ONTOLOGY}"
echo "  Expert Expanded dir: ${EXPERT_EXPANDED_DIR}"
echo "  LLM Expanded dir   : ${LLM_EXPANDED_DIR}"
echo "  Figures dir        : ${OUTPUT_DIR}"

python3 "${MODULE_ROOT}/scripts/07.figure_no_quantization.py" \
    --expert_expanded_dir "${EXPERT_EXPANDED_DIR}" \
    --expert_orig_dir     "${EXPERT_ORIG_DIR}" \
    --llm_expanded_dir    "${LLM_EXPANDED_DIR}" \
    --output_dir           "${OUTPUT_DIR}" \
    --ontology             "${ONTOLOGY}"

echo "Done. Table written to: ${OUTPUT_DIR}/quantization_comparison_table.png"
