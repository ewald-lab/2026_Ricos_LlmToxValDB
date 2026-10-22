#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --mem=80G
#SBATCH -J "toxval_llm_production_run"
#SBATCH -o "logs/out_tvllm_prod_%j.out"
#SBATCH -e "logs/err_tvllm_prod_%j.err"

# Arguments: endpoint (required), model (optional), port (optional), ontology (optional)
ENDPOINT="${1:-}"
MODEL="${2:-Qwen3.6-35B-A3B-FP8}"
PORT="${3:-8053}"
ONTOLOGY="${4:-mp}"

if [ -z "$ENDPOINT" ]; then
    echo "Error: Host node / endpoint is required."
    echo "Usage: sbatch $0 <codon-node|base_url> [model] [port] [ontology]"
    echo "Example: sbatch $0 codon-gpu-015 Qwen3.6-35B-A3B-FP8 8053 mp"
    exit 1
fi

if [[ "$ENDPOINT" =~ ^https?:// ]]; then
    BASE_URL="$ENDPOINT"
else
    BASE_URL="http://${ENDPOINT}.ebi.ac.uk:${PORT}/v1"
fi

CURRENT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$CURRENT_DIR"
while [ "$REPO_ROOT" != "/" ] && [ ! -d "$REPO_ROOT/HiPO" ]; do
    REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
done

if [ ! -d "$REPO_ROOT/HiPO" ]; then
    echo "Error: Could not locate repository root containing HiPO directory."
    exit 1
fi

export TOXVALDB_ROOT="${REPO_ROOT}/toxvaldb"
HIPO_DIR="${REPO_ROOT}/HiPO"
MODULE_ROOT="${TOXVALDB_ROOT}/01.hipo/modules/00.hipo_toxvaldb"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-/nfs/research/ewald/ricos/huggingface_cache}"

module purge && module load cuda/12.8.0 gcc

VENV_DIR="${TOXVALDB_ROOT}/01.hipo/env-text-to-onto"
source "${VENV_DIR}/bin/activate"

mkdir -p "${MODULE_ROOT}/jobs/logs"
mkdir -p "${MODULE_ROOT}/outputs"

"${VENV_DIR}/bin/python" "${HIPO_DIR}/hipo.py" \
    --input_file "${TOXVALDB_ROOT}/00.data_preparation/modules/01.preprocess_toxvaldb/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv" \
    --output_file "${MODULE_ROOT}/outputs/ToxValDB_Classified_${ONTOLOGY}_${MODEL}.csv" \
    --reasoning_file "${MODULE_ROOT}/outputs/ToxValDB_Reasoning_${ONTOLOGY}_${MODEL}.json" \
    --target_column TOXICOLOGICAL_EFFECT_LLM_CLEANED \
    --system_prompt_file "${MODULE_ROOT}/config/system_prompt_{MODE}.txt" \
    --rag_config "${MODULE_ROOT}/inputs/choice_rag.json" \
    --ontology_embedding_db "${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/{EMBED_DIR}/${ONTOLOGY}.db" \
    --global_definitions_db "${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/definitions_${ONTOLOGY}.db" \
    --target_embedding_db "${TOXVALDB_ROOT}/00.data_preparation/modules/01.preprocess_toxvaldb/outputs/embeddings_db/tox_effects_{EMBED_DIR}.db" \
    --base_url "${BASE_URL}" \
    --num_backups 10 \
    --context_ontology_file "${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt" \
    --model "${MODEL}" \
    --llm_config "${MODULE_ROOT}/inputs/choice_llm.json" \
    --descriptions_concurrency 100 \
    --ontology_definitions
