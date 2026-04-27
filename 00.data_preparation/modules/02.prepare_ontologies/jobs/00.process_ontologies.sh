#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G
#SBATCH -J "toxval_llm_flow"
#SBATCH -o "logs/out_hierarchy_%A_%a.out"
#SBATCH -e "logs/err_hierarchy_%A_%a.err"
#SBATCH --array=0-19

ONTOLOGIES=("hp" "mp" "aop" "hpath" "txpo")
LLM_MODEL="gpt-oss-120b"

MODELS_KEYS=("qwen" "nemotron" "harrier" "kalm")
declare -A MODELS
MODELS=(
    ["qwen"]="Qwen/Qwen3-Embedding-8B"
    ["nemotron"]="nvidia/llama-embed-nemotron-8b"
    ["harrier"]="microsoft/harrier-oss-v1-27b"
    ["kalm"]="tencent/KaLM-Embedding-Gemma3-12B-2511"
)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
PREP_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge
module load cuda/12.8.0
module load gcc

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH

# --- Environment Setup ---
source "${PREP_ROOT}/env-data-prep/bin/activate"

export PYTHONPATH="${REPO_ROOT}/analysis:$PYTHONPATH"

TASK_ID=$SLURM_ARRAY_TASK_ID
NUM_MODELS=${#MODELS_KEYS[@]}

ONTOLOGY_IDX=$(( TASK_ID / NUM_MODELS ))
MODEL_IDX=$(( TASK_ID % NUM_MODELS ))

ONTOLOGY="${ONTOLOGIES[$ONTOLOGY_IDX]}"
EMBED_DIR="${MODELS_KEYS[$MODEL_IDX]}"
EMBED_MODEL="${MODELS[$EMBED_DIR]}"

RAG_CONFIG="${REPO_ROOT}/config/rag/embeddings.json"

echo "=========================================================="
echo "Task ID: $TASK_ID | Processing Ontology: $ONTOLOGY | Model: $EMBED_DIR"
echo "=========================================================="

# Sync flag file unique per ontology
SYNC_FILE="${REPO_ROOT}/outputs/${ONTOLOGY}_definitions_done.flag"

if [ $MODEL_IDX -eq 0 ]; then
    echo "First task ($EMBED_DIR). Processing definitions..."

    ROOT_FLAG=""
    if [ "$ONTOLOGY" == "hp" ]; then
        ROOT_FLAG="--root_uri http://purl.obolibrary.org/obo/HP_0000118"
    elif [ "$ONTOLOGY" == "mp" ]; then
        ROOT_FLAG="--root_uri http://purl.obolibrary.org/obo/MP_0000001"
    fi

    python "${REPO_ROOT}/scripts/01.process_ontology.py" \
        $ROOT_FLAG \
        --ontology_file "${REPO_ROOT}/inputs/owl_files/${ONTOLOGY}.ttl" \
        --ontology_db "${REPO_ROOT}/inputs/ontology_dbs/${EMBED_DIR}/${ONTOLOGY}.db" \
        --rag_config "${RAG_CONFIG}" \
        --embeddings_model_id "${EMBED_DIR}" \
        --global_definitions_db "${REPO_ROOT}/inputs/ontology_dbs/definitions_${ONTOLOGY}_${LLM_MODEL}.db" \
        --prompt_template "${REPO_ROOT}/config/tools/create_description_prompt.txt" \
        --model_config "${REPO_ROOT}/config/models/${LLM_MODEL}.json" \
        --base_url "http://codon-gpu-004.ebi.ac.uk:8004/v1"

    if [ $? -eq 0 ]; then
        touch "$SYNC_FILE"
        echo "Definitions processed. Flag file created."
    else
        echo "Error processing definitions. Aborting."
        exit 1
    fi

else
    echo "Waiting for first task to finish processing definitions..."

    while [ ! -f "$SYNC_FILE" ]; do
        sleep 60
    done

    ROOT_FLAG=""
    if [ "$ONTOLOGY" == "hp" ]; then
        ROOT_FLAG="--root_uri http://purl.obolibrary.org/obo/HP_0000118"
    elif [ "$ONTOLOGY" == "mp" ]; then
        ROOT_FLAG="--root_uri http://purl.obolibrary.org/obo/MP_0000001"
    fi

    python "${REPO_ROOT}/scripts/01.process_ontology.py" \
        $ROOT_FLAG \
        --ontology_file "${REPO_ROOT}/inputs/owl_files/${ONTOLOGY}.ttl" \
        --ontology_db "${REPO_ROOT}/inputs/ontology_dbs/${EMBED_DIR}/${ONTOLOGY}.db" \
        --rag_config "${RAG_CONFIG}" \
        --embeddings_model_id "${EMBED_DIR}" \
        --global_definitions_db "${REPO_ROOT}/inputs/ontology_dbs/definitions_${ONTOLOGY}_${LLM_MODEL}.db" \
        --prompt_template "${REPO_ROOT}/config/tools/create_description_prompt.txt" \
        --model_config "${REPO_ROOT}/config/models/${LLM_MODEL}.json" \
        --base_url "http://codon-gpu-004.ebi.ac.uk:8004/v1"
fi
