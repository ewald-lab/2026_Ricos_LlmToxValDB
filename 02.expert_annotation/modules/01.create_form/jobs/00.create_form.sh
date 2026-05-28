#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH -J "expert_html_gen"
#SBATCH -o "logs/out_expert_gen_%j.out"
#SBATCH -e "logs/err_expert_gen_%j.err"

ONTOLOGY="hpath"
TOP_CANDIDATES=10
NUM_PHRASES=50

JOBS_DIR="$SLURM_SUBMIT_DIR"
MODULE_ROOT="$(cd "${JOBS_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"
GIT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

module purge
module load gcc

source "${PROJECT_ROOT}/env-expert-annot/bin/activate"


python "${MODULE_ROOT}/scripts/01.expert_test.py" \
    --input_pool "${GIT_ROOT}/02.expert_annotation/modules/00.create_test/outputs/test_pool_${ONTOLOGY}.json" \
    --ontology_db "${GIT_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/qwen/${ONTOLOGY}.db" \
    --output_file "${MODULE_ROOT}/outputs/expert_annotation_${ONTOLOGY}.html" \
    --email "ricos@ebi.ac.uk" \
    --context_ontology "$(cat "${GIT_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt")" \
    --top_reranker ${TOP_CANDIDATES} \
    --num_phrases ${NUM_PHRASES} \
    --skip_first_level \
    #--debug
