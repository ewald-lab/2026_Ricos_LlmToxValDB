#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH -J "eval_results"
#SBATCH --array=0
#SBATCH -o "logs/out_eval_results_%A_%a.out"
#SBATCH -e "logs/err_eval_results_%A_%a.err"

# Ensure logs directory exists
mkdir -p logs

if [ -n "${SLURM_SUBMIT_DIR}" ] && [ -f "${SLURM_SUBMIT_DIR}/03.evaluation/modules/03.eval_results/jobs/run_eval_results.sh" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
    SCRIPT_DIR="${REPO_ROOT}/03.evaluation/modules/03.eval_results/jobs"
else
    SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
fi

ONTOLOGIES=("mp" "aop" "txpo" "hpath" "hp")
ROOT_TERMS=(
    "mammalian phenotype"
    "adverse outcome pathway"
    "toxic course"
    "morphologic change"
    "phenotypic abnormality"
)
TREE_MIN_CASES=(500 100 500 500 500)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ONTOLOGY="${ONTOLOGIES[$TASK_ID]}"
ROOT_TERM="${ROOT_TERMS[$TASK_ID]}"
TREE_MIN_CASE="${TREE_MIN_CASES[$TASK_ID]}"

if [ -z "${ONTOLOGY}" ] || [ -z "${ROOT_TERM}" ] || [ -z "${TREE_MIN_CASE}" ]; then
    echo "ERROR: Invalid SLURM_ARRAY_TASK_ID='${TASK_ID}'." >&2
    exit 1
fi

INPUT_FILE="${REPO_ROOT}/01.text_to_onto/modules/00.tox_effect_to_onto/outputs/ToxValDB_Classified_${ONTOLOGY}_Qwen3.6-35B-A3B-FP8.csv"
OWL_FILE="${REPO_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/owl_files/${ONTOLOGY}.ttl"
JSON_FILE="${REPO_ROOT}/01.text_to_onto/modules/00.tox_effect_to_onto/outputs/ToxValDB_Reasoning_${ONTOLOGY}_Qwen3.6-35B-A3B-FP8.json"
ONTOLOGY_DB="${REPO_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/qwen/${ONTOLOGY}.db"
FIGURES_DIR="${REPO_ROOT}/03.evaluation/modules/03.eval_results/outputs/${ONTOLOGY}"

source "${REPO_ROOT}/03.evaluation/env-evaluation/bin/activate"

mkdir -p "${FIGURES_DIR}"

echo "=== Evaluating results for ontology '${ONTOLOGY}' ==="
echo "  Array task : ${TASK_ID}"
echo "  Root term  : ${ROOT_TERM}"
echo "  Input CSV  : ${INPUT_FILE}"
echo "  JSON       : ${JSON_FILE}"
echo "  OWL        : ${OWL_FILE}"
echo "  DB         : ${ONTOLOGY_DB}"
echo "  Output dir : ${FIGURES_DIR}"
echo "  Tree min cases: ${TREE_MIN_CASE}"

if [ -s "${FIGURES_DIR}/year_depth_stats.csv" ]; then
    echo "Year depth stats cache exists; skipping CSV depth analysis."
else
    python "${REPO_ROOT}/03.evaluation/modules/03.eval_results/scripts/eval_results.py" \
        --input_file  "${INPUT_FILE}" \
        --json_file   "${JSON_FILE}" \
        --owl_file    "${OWL_FILE}" \
        --output_dir  "${FIGURES_DIR}" \
        --root_term   "${ROOT_TERM}" \
        --threshold   500 \
        --depth_only
fi


EXTRA_ARGS=""
if [ "${ONTOLOGY}" = "aop" ] || [ "${ONTOLOGY}" = "txpo" ]; then
    EXTRA_ARGS="--no_inner_donut"
fi
if [ "${ONTOLOGY}" = "hpath" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --no_inner_label_2"
fi

python "${REPO_ROOT}/03.evaluation/modules/03.eval_results/scripts/gen_mapping_success_depth.py" \
    --json_file   "${JSON_FILE}" \
    --ontology_db "${ONTOLOGY_DB}" \
    --output_dir  "${FIGURES_DIR}" \
    --root_term   "${ROOT_TERM}" \
    ${EXTRA_ARGS}

echo "Mapping origin success summary done. Now generating phenotypic landscape..."

python "${REPO_ROOT}/03.evaluation/modules/03.eval_results/scripts/gen_phenotypic_landscape.py" \
    --json_file   "${JSON_FILE}" \
    --ontology_db "${ONTOLOGY_DB}" \
    --output_dir  "${FIGURES_DIR}" \
    --root_term   "${ROOT_TERM}"

echo "Phenotypic landscape done. Now generating ontology tree..."

python "${REPO_ROOT}/03.evaluation/modules/03.eval_results/scripts/gen_ontology_tree.py" \
    --json_file   "${JSON_FILE}" \
    --owl_file    "${OWL_FILE}" \
    --output_dir  "${FIGURES_DIR}" \
    --root_term   "${ROOT_TERM}" \
    --threshold   "${TREE_MIN_CASE}"

echo "Done."
