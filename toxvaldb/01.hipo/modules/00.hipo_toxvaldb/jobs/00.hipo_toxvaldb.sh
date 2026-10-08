#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH -J "toxval_llm_production_run"
#SBATCH -o "logs/out_tvllm_prod_%j.out"
#SBATCH -e "logs/err_tvllm_prod_%j.err"

# Arguments: endpoint (required), model (optional), port (optional), ontology (optional)
ENDPOINT="${1:-}"
MODEL="${2:-Qwen3.6-35B-A3B-FP8}"
PORT="${3:-8052}"
ONTOLOGY="${4:-hpath}"

if [ -z "$ENDPOINT" ]; then
    echo "Error: Host node / endpoint is required."
    echo "Usage: sbatch $0 <codon-node|base_url> [model] [port] [ontology]"
    echo "Example: sbatch $0 codon-slurm-login-01 Qwen3.6-35B-A3B-FP8 8052 hpath"
    exit 1
fi

if [[ "$ENDPOINT" =~ ^https?:// ]]; then
    BASE_URL="$ENDPOINT"
else
    BASE_URL="http://${ENDPOINT}.ebi.ac.uk:${PORT}/v1"
fi

if [[ "${BASH_SOURCE[0]}" == /var/spool/slurmd/* ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export TOXVALDB_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
REPO_ROOT="$(cd "${TOXVALDB_ROOT}/.." && pwd)"
HIPO_DIR="${REPO_ROOT}/HiPO"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge && module load cuda/12.8.0 gcc

source "${TOXVALDB_ROOT}/01.hipo/env-text-to-onto/bin/activate"

python "${HIPO_DIR}/hipo.py" \
    --input_file "${TOXVALDB_ROOT}/00.data_preparation/modules/01.preprocess_toxvaldb/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv" \
    --output_file "${TOXVALDB_ROOT}/01.hipo/modules/00.hipo_toxvaldb/outputs/ToxValDB_Classified_${ONTOLOGY}_${MODEL}.csv" \
    --reasoning_file "${TOXVALDB_ROOT}/01.hipo/modules/00.hipo_toxvaldb/outputs/ToxValDB_Reasoning_${ONTOLOGY}_${MODEL}.json" \
    --target_column TOXICOLOGICAL_EFFECT_LLM_CLEANED \
    --system_prompt_file "${TOXVALDB_ROOT}/01.hipo/modules/00.hipo_toxvaldb/config/system_prompt_{MODE}.txt" \
    --rag_config "${TOXVALDB_ROOT}/01.hipo/modules/00.hipo_toxvaldb/inputs/choice_rag.json" \
    --ontology_embedding_db "${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/{EMBED_DIR}/${ONTOLOGY}.db" \
    --global_definitions_db "${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/definitions_${ONTOLOGY}.db" \
    --target_embedding_db "${TOXVALDB_ROOT}/00.data_preparation/modules/01.preprocess_toxvaldb/outputs/embeddings_db/tox_effects_{EMBED_DIR}.db" \
    --base_url "${BASE_URL}" \
    --num_backups 10 \
    --context_ontology_file "${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt" \
    --model "${MODEL}" \
    --llm_config "${TOXVALDB_ROOT}/01.hipo/modules/00.hipo_toxvaldb/inputs/choice_llm.json" \
    --descriptions_concurrency 100 \
    --ontology_definitions
