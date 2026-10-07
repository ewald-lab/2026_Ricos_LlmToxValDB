#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "eval_unmapped"
#SBATCH --array=0
#SBATCH -o "logs/out_unmapped_%A_%a.out"
#SBATCH -e "logs/err_unmapped_%A_%a.err"

# Ensure logs directory exists
mkdir -p logs

if [ -n "${SLURM_SUBMIT_DIR}" ] && [ -f "${SLURM_SUBMIT_DIR}/toxvaldb/03.evaluation/modules/03.eval_results/jobs/run_unmapped_analysis.sh" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}/toxvaldb"
elif [ -n "${SLURM_SUBMIT_DIR}" ] && [ -f "${SLURM_SUBMIT_DIR}/03.evaluation/modules/03.eval_results/jobs/run_unmapped_analysis.sh" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
    SCRIPT_DIR="${REPO_ROOT}/03.evaluation/modules/03.eval_results/jobs"
else
    SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
fi

ONTOLOGIES=("mp" "aop" "txpo" "hpath" "hp")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ONTOLOGY="${ONTOLOGIES[$TASK_ID]}"

if [ -z "${ONTOLOGY}" ]; then
    echo "ERROR: Invalid SLURM_ARRAY_TASK_ID='${TASK_ID}'." >&2
    exit 1
fi

JSON_FILE="${REPO_ROOT}/01.text_to_onto/modules/00.tox_effect_to_onto/outputs/ToxValDB_Reasoning_${ONTOLOGY}_Qwen3.6-35B-A3B-FP8.json"
OUTPUT_DIR="${REPO_ROOT}/03.evaluation/modules/03.eval_results/outputs/${ONTOLOGY}"

source "${REPO_ROOT}/03.evaluation/env-evaluation/bin/activate"

mkdir -p "${OUTPUT_DIR}"

echo "=== Analyzing unmapped phrases for ontology '${ONTOLOGY}' ==="
echo "  Array task : ${TASK_ID}"
echo "  JSON file  : ${JSON_FILE}"
echo "  Output dir : ${OUTPUT_DIR}"

python "${REPO_ROOT}/03.evaluation/modules/03.eval_results/scripts/analyze_unmapped_phrases.py" \
    --json_file  "${JSON_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --ontology   "${ONTOLOGY}"

echo "Unmapped analysis completed successfully."
