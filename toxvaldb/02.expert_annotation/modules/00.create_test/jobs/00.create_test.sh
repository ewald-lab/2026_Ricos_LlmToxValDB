#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -J "create_test"
#SBATCH -o "logs/out_create_test_%j.out"
#SBATCH -e "logs/err_create_test_%j.err"

# Arguments: endpoint (required), model (optional), port (optional)
ENDPOINT="${1:-${CODON_NODE:-}}"
MODEL="${2:-Qwen3.6-35B-A3B-FP8}"
PORT="${3:-8052}"

if [ -z "$ENDPOINT" ]; then
    echo "Error: Host node / endpoint is required. Pass it as the first argument or set CODON_NODE."
    echo "Usage: sbatch $0 <codon-node|base_url> [model] [port]"
    echo "Example: sbatch $0 codon-gpu-010 Qwen3.6-35B-A3B-FP8 8052"
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

MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERT_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"
TOXVALDB_ROOT="$(cd "${EXPERT_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${TOXVALDB_ROOT}/.." && pwd)"
HIPO_DIR="${REPO_ROOT}/HiPO"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge && module load cuda/12.8.0 gcc

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
if [ -d "$GCC_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH
fi

source "${EXPERT_ROOT}/env-expert-annot/bin/activate"

mkdir -p "${MODULE_ROOT}/logs" "${MODULE_ROOT}/outputs" "${MODULE_ROOT}/figures"

for ONTOLOGY in hpath mp; do
    echo "=== Processing ontology: ${ONTOLOGY} ==="

    EXTRA_FLAGS=""
    if [ "${ONTOLOGY}" = "hpath" ]; then
        EXTRA_FLAGS="--histopathology"
    fi

    python "${MODULE_ROOT}/scripts/00.create_test.py" \
        --input_file "${TOXVALDB_ROOT}/00.data_preparation/modules/01.preprocess_toxvaldb/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv" \
        --output_file "${MODULE_ROOT}/outputs/test_candidates_${ONTOLOGY}.txt" \
        --output_json "${MODULE_ROOT}/outputs/test_pool_${ONTOLOGY}.json" \
        --target_column TOXICOLOGICAL_EFFECT_LLM_CLEANED \
        --rag_config "${HIPO_DIR}/config/choice_rag.json" \
        --target_embedding_db "${TOXVALDB_ROOT}/00.data_preparation/modules/01.preprocess_toxvaldb/outputs/embeddings_db/tox_effects_qwen.db" \
        --ontology_embedding_db "${TOXVALDB_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/qwen/${ONTOLOGY}.db" \
        --output_visualization "${MODULE_ROOT}/figures/test_visualization_${ONTOLOGY}.png" \
        --num_samples 50 \
        --seed 2011 \
        --model_config "${HIPO_DIR}/config/llm.json" \
        --llm_model "${MODEL}" \
        --base_url "${BASE_URL}" \
        --system_prompt_file "${TOXVALDB_ROOT}/01.hipo/modules/00.hipo_toxvaldb/config/system_prompt_1.txt" \
        ${EXTRA_FLAGS}

done
