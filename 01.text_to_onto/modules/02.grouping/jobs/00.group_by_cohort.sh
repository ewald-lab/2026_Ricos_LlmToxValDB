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

JOBS_DIR="$SLURM_SUBMIT_DIR"
REPO_ROOT="$(cd "${JOBS_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"
GIT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

TOX_EFFECT_MODULE="${PROJECT_ROOT}/modules/00.tox_effect_to_onto"
GUIDELINE_MODULE="${PROJECT_ROOT}/modules/01.guideline_to_onto"

source "${PROJECT_ROOT}/env-text-to-onto/bin/activate"

python "${REPO_ROOT}/scripts/00.group_by_cohort.py" \
    --input_file "${TOX_EFFECT_MODULE}/outputs/ToxValDB_Classified_${ONTOLOGY}_${MODEL}.csv" \
    --guidelines_dir "${GUIDELINE_MODULE}/outputs/guidelines_mapping/${ONTOLOGY}/guidelines" \
    --output_file "${REPO_ROOT}/outputs/ToxValDB_Cohort_${ONTOLOGY}_${MODEL}.csv" \
    --concentration_col TOXVAL_NUMERIC \
    --toxval_type_col TOXVAL_TYPE \
    --negative_col NEGATIVE \
    --guideline_col MATRIX_GUIDELINE
