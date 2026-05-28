#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "toxval_group_cohort"
#SBATCH -o "logs/out_group_cohort_%j.out"
#SBATCH -e "logs/err_group_cohort_%j.err"

ONTOLOGY="mp"
MODEL="Qwen3.5-122B-A10B-FP8"

REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"

source "${PROJECT_ROOT}/env-text-to-onto/bin/activate"

python "${REPO_ROOT}/scripts/00.group_by_cohort.py" \
    --input_file "${PROJECT_ROOT}/modules/00.tox_effect_to_onto/outputs/ToxValDB_Classified_${ONTOLOGY}_${MODEL}.csv" \
    --guidelines_dir "${PROJECT_ROOT}/modules/01.guideline_to_onto/outputs/guidelines_mapping/${ONTOLOGY}/guidelines" \
    --output_file "${REPO_ROOT}/outputs/ToxValDB_Cohort_${ONTOLOGY}_${MODEL}.csv" \
    --concentration_col TOXVAL_NUMERIC \
    --toxval_type_col TOXVAL_TYPE \
    --negative_col NEGATIVE \
    --guideline_col MATRIX_GUIDELINE
