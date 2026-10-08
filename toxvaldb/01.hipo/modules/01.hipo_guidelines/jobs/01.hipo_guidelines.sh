#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=25G
#SBATCH -J "endpoint_to_phenotype"
#SBATCH --array=0-4
#SBATCH -o "logs/out_endpoint_to_phenotype_%A_%a.out"
#SBATCH -e "logs/err_endpoint_to_phenotype_%A_%a.err"

ONTOLOGIES=(mp hpath aop hp txpo)

if [[ "${BASH_SOURCE[0]}" == /var/spool/slurmd/* ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HIPO_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"
DATA_PREP_ROOT="$(cd "${HIPO_ROOT}/../00.data_preparation" && pwd)"

ENDPOINT="${1:-${CODON_NODE:-}}"
PORT="${2:-8052}"
MODEL="${3:-Qwen3.6-35B-A3B-FP8}"
ONTOLOGY_ARG="${4:-hpath}"

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    ONTOLOGY="${ONTOLOGIES[$SLURM_ARRAY_TASK_ID]}"
else
    ONTOLOGY="$ONTOLOGY_ARG"
fi

if [ -z "$ENDPOINT" ]; then
    echo "Error: Host node / endpoint is required."
    echo "Usage: sbatch $0 <codon-node|base_url> [port] [model] [ontology]"
    echo "Example: sbatch $0 codon-gpu-010 8052 Qwen3.6-35B-A3B-FP8 hpath"
    exit 1
fi

if [[ "$ENDPOINT" =~ ^https?:// ]]; then
    BASE_URL="$ENDPOINT"
else
    BASE_URL="http://${ENDPOINT}.ebi.ac.uk:${PORT}/v1"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-/nfs/research/ewald/ricos/huggingface_cache}"

module purge && module load cuda/12.8.0 gcc

source "${HIPO_ROOT}/env-text-to-onto/bin/activate"

mkdir -p "${MODULE_ROOT}/outputs"
mkdir -p "${SCRIPT_DIR}/logs"

echo "--- Processing Ontology: ${ONTOLOGY} (Task ID: ${SLURM_ARRAY_TASK_ID:-single}) ---"

python "${MODULE_ROOT}/scripts/00.guideline_to_onto.py"     --input_endpoints "${DATA_PREP_ROOT}/modules/03.prepare_guidelines/outputs/normalized_endpoint_list.csv"     --output_dir "${MODULE_ROOT}/outputs"     --ontology "${ONTOLOGY}"     --ontology_embedding_db "${DATA_PREP_ROOT}/modules/02.prepare_ontologies/outputs/{EMBED_DIR}/${ONTOLOGY}.db"     --global_definitions_db "${DATA_PREP_ROOT}/modules/02.prepare_ontologies/outputs/definitions_${ONTOLOGY}.db"     --base_url "${BASE_URL}"     --system_prompt_file "${MODULE_ROOT}/config/system_prompt_endpoint_filter.txt"     --context_ontology "${DATA_PREP_ROOT}/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt"     --rag_config "${MODULE_ROOT}/config/rag/best_config.json"     --llm_config "${MODULE_ROOT}/config/llm_config.json"     --model "${MODEL}"
