#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH -J "average_human"
#SBATCH -o "logs/average_human_%j.out"
#SBATCH -e "logs/average_human_%j.err"

if [[ "${BASH_SOURCE[0]}" == /var/spool/slurmd/* ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"
TOXVALDB_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${TOXVALDB_ROOT}/.." && pwd)"

mkdir -p "${SCRIPT_DIR}/logs" "${MODULE_ROOT}/outputs"

INPUT_DIR="${MODULE_ROOT}/outputs"
OUTPUT_FILE="${MODULE_ROOT}/outputs/Expanded_average_human.csv"
MIN_EXPERTS="${1:-2}"

module purge 2>/dev/null || true
module load gcc 2>/dev/null || true

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
if [ -d "$GCC_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH
fi

source "${EVAL_ROOT}/env-evaluation/bin/activate"

echo "=== Creating average human profile ==="
echo "  Input dir  : ${INPUT_DIR}"
echo "  Output file: ${OUTPUT_FILE}"
echo "  Min experts: ${MIN_EXPERTS}"

python3 "${MODULE_ROOT}/scripts/average_human.py"     --input_dir   "${INPUT_DIR}"     --output_file "${OUTPUT_FILE}"     --min_experts "${MIN_EXPERTS}"

echo "Done."
