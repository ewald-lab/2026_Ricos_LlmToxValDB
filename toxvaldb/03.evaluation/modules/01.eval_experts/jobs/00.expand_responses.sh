#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "expand_expert_responses"
#SBATCH -o "logs/expand_responses_%j.out"
#SBATCH -e "logs/expand_responses_%j.err"

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

ONTOLOGY="${1:-mp}"
ROOT_TERM="${2:-mammalian phenotype}"

INPUT_DIR="${MODULE_ROOT}/inputs/${ONTOLOGY}"
OWL_FILE="${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/owl_files/${ONTOLOGY}.ttl"
OUTPUT_DIR="${MODULE_ROOT}/outputs"

module purge 2>/dev/null || true
module load gcc 2>/dev/null || true

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
if [ -d "$GCC_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH
fi

source "${EVAL_ROOT}/env-evaluation/bin/activate"

echo "=== Expanding expert responses ==="
echo "  Ontology  : ${ONTOLOGY}"
echo "  OWL file  : ${OWL_FILE}"
echo "  Input dir : ${INPUT_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Root term : ${ROOT_TERM}"

python3 "${MODULE_ROOT}/scripts/run_analysis.py"     --input_dir  "${INPUT_DIR}"     --owl_file   "${OWL_FILE}"     --output_dir "${OUTPUT_DIR}"     --ontology   "${ONTOLOGY}"     --root_term  "${ROOT_TERM}"

echo "Done. Expanded files written to: ${OUTPUT_DIR}"
