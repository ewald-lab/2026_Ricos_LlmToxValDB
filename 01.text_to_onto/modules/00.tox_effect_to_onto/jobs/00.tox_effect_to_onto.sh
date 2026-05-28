#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=80G
#SBATCH -J "toxval_llm_production_run"
#SBATCH -o "logs/out_tvllm_prod_%j.out"
#SBATCH -e "logs/err_tvllm_prod_%j.err"

CODON_NODE="${1:-${CODON_NODE:-}}"
if [ -z "$CODON_NODE" ]; then
    echo "Error: CODON_NODE is required. Pass it as the first argument or set the CODON_NODE env variable."
    echo "Usage: sbatch $0 <codon-node>  (e.g. sbatch $0 codon-gpu-001)"
    exit 1
fi

ONTOLOGY="mp"

BASE_URL="http://${CODON_NODE}.ebi.ac.uk:PORT/v1"
REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/../../../../" && pwd)"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge && module load cuda/12.8.0 gcc

source "${REPO_ROOT}/01.text_to_onto/env-text-to-onto/bin/activate"

python "${REPO_ROOT}/01.text_to_onto/modules/00.tox_effect_to_onto/scripts/00.tox_effect_to_onto.py" \
    --input_file "${REPO_ROOT}/00.data_preparation/modules/01.prepare_database/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv" \
    --output_file "${REPO_ROOT}/01.text_to_onto/modules/00.tox_effect_to_onto/outputs/ToxValDB_Classified_${ONTOLOGY}_{MODEL}.csv" \
    --reasoning_file "${REPO_ROOT}/01.text_to_onto/modules/00.tox_effect_to_onto/outputs/ToxValDB_Reasoning_${ONTOLOGY}_{MODEL}.json" \
    --target_column TOXICOLOGICAL_EFFECT_LLM_CLEANED \
    --std_column TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED \
    --system_prompt_file "${REPO_ROOT}/01.text_to_onto/modules/00.tox_effect_to_onto/config/system_prompt_{MODE}.txt" \
    --rag_config "${REPO_ROOT}/01.text_to_onto/modules/00.tox_effect_to_onto/config/rag/best_config.json" \
    --ontology_embedding_db "${REPO_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/{EMBED_DIR}/${ONTOLOGY}.db" \
    --global_definitions_db "${REPO_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/definitions_${ONTOLOGY}.db" \
    --target_embedding_db "${REPO_ROOT}/00.data_preparation/modules/01.prepare_database/outputs/embeddings_db/tox_effects_{EMBED_DIR}.db" \
    --base_url "${BASE_URL}" \
    --num_backups 10 \
    --context_ontology "$(cat "${REPO_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt")" \
    --model_config "placeholder" \
    --llm_config "${REPO_ROOT}/01.text_to_onto/modules/00.tox_effect_to_onto/config/llm_config.json" \
    --descriptions_concurrency 100
