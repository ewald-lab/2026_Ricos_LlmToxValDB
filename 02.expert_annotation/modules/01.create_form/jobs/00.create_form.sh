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
REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/../../../../" && pwd)"

module purge && module load gcc

source "${REPO_ROOT}/02.expert_annotation/env-expert-annot/bin/activate"

python "${REPO_ROOT}/02.expert_annotation/modules/01.create_form/scripts/01.expert_test.py" \
    --input_pool "${REPO_ROOT}/02.expert_annotation/modules/00.create_test/outputs/test_pool_${ONTOLOGY}.json" \
    --ontology_db "${REPO_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/qwen/${ONTOLOGY}.db" \
    --global_definitions_db "${REPO_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/definitions_${ONTOLOGY}.db" \
    --output_file "${REPO_ROOT}/02.expert_annotation/modules/01.create_form/outputs/expert_annotation_${ONTOLOGY}.html" \
    --email "ricos@ebi.ac.uk" \
    --context_ontology "$(cat "${REPO_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt")" \
    --top_reranker 10 \
    --num_phrases 50 \
    --skip_first_level
