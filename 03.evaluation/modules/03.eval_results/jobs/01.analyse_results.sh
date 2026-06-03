#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH -J "analyse_missing_mp"
#SBATCH -o "logs/out_analyse_missing_%j.out"
#SBATCH -e "logs/err_analyse_missing_%j.err"

# Ensure logs directory exists
mkdir -p logs

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
if [ -z "${SCRIPT_DIR}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

ONTOLOGY="mp"
ROOT_TERM="mammalian phenotype"

INPUT_FILE="${REPO_ROOT}/01.text_to_onto/modules/00.tox_effect_to_onto/outputs/ToxValDB_Classified_${ONTOLOGY}_Qwen3.6-35B-A3B-FP8.csv"
OUTPUT_FILE="${REPO_ROOT}/01.text_to_onto/modules/00.tox_effect_to_onto/outputs/missing_analysis_${ONTOLOGY}_Qwen3.6-35B-A3B-FP8.tsv"
OWL_FILE="${REPO_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/owl_files/${ONTOLOGY}.ttl"
FIGURES_DIR="${REPO_ROOT}/03.evaluation/modules/03.eval_results/outputs"

source "${REPO_ROOT}/03.evaluation/env-evaluation/bin/activate"

python "${REPO_ROOT}/03.evaluation/modules/03.eval_results/scripts/01.plot_year_vs_depth.py" \
    --input_file  "${INPUT_FILE}" \
    --owl_file    "${OWL_FILE}" \
    --output_dir  "${FIGURES_DIR}" \
    --root_term   "${ROOT_TERM}"

echo "Done."
