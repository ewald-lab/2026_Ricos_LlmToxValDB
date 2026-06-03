#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=80G
#SBATCH -J "endpoint_to_phenotype"
#SBATCH --array=0-4
#SBATCH -o "logs/out_endpoint_to_phenotype_%A_%a.out"
#SBATCH -e "logs/err_endpoint_to_phenotype_%A_%a.err"

ONTOLOGIES=(mp hpath aop hp txpo)

MODULE_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
DATA_PREP_ROOT="$(cd "${MODULE_ROOT}/../../00.data_preparation" && pwd)"

CODON_NODE="${1:-${CODON_NODE:-hl-codon-gpu-020}}"
BASE_URL="http://${CODON_NODE}.ebi.ac.uk:PORT/v1"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-/nfs/research/ewald/ricos/huggingface_cache}"

module purge && module load cuda/12.8.0 gcc

source "${DATA_PREP_ROOT}/env-data-prep/bin/activate"

mkdir -p "${MODULE_ROOT}/outputs"

ONTOLOGY="${ONTOLOGIES[$SLURM_ARRAY_TASK_ID]}"
echo "--- Processing Ontology: ${ONTOLOGY} (Task ID: ${SLURM_ARRAY_TASK_ID}) ---"

python "${MODULE_ROOT}/scripts/00.guideline_to_onto.py" \
    --input_endpoints "${DATA_PREP_ROOT}/modules/03.prepare_guidelines/outputs/normalized_endpoint_list.csv" \
    --output_dir "${MODULE_ROOT}/outputs" \
    --ontology "${ONTOLOGY}" \
    --ontology_embedding_db "${DATA_PREP_ROOT}/modules/02.prepare_ontologies/outputs/{EMBED_DIR}/${ONTOLOGY}.db" \
    --global_definitions_db "${DATA_PREP_ROOT}/modules/02.prepare_ontologies/outputs/definitions_${ONTOLOGY}.db" \
    --base_url "${BASE_URL}" \
    --system_prompt_file "${MODULE_ROOT}/config/system_prompt_endpoint_filter.txt" \
    --context_ontology "${DATA_PREP_ROOT}/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt" \
    --rag_config "${MODULE_ROOT}/config/rag/best_config.json" \
    --model_config "placeholder" \
    --llm_config "${MODULE_ROOT}/config/llm_config.json"
