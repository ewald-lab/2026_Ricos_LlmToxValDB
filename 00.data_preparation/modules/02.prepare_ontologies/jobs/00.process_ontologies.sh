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
LLM_MODEL="Qwen3.5-122B-A10B-FP8"
MODEL_CONFIG="/nfs/research/ewald/ricos/code/playground/models/config/${LLM_MODEL}.json"
CODON_NODE="${1:-${CODON_NODE:-}}"
if [ -z "$CODON_NODE" ]; then
    echo "Error: CODON_NODE is required. Pass it as the first argument or set the CODON_NODE env variable."
    echo "Usage: sbatch $0 <codon-node>  (e.g. sbatch $0 codon-gpu-004)"
    exit 1
fi
BASE_URL="http://${CODON_NODE}.ebi.ac.uk:8003/v1"

MODELS_KEYS=("qwen" "nemotron" "harrier" "kalm")

SCRIPT_PATH=$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

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
        --global_definitions_db "${REPO_ROOT}/inputs/ontology_dbs/definitions_${ONTOLOGY}.db" \
        --prompt_template "${REPO_ROOT}/config/tools/create_description_prompt.txt" \
        --model_config "${MODEL_CONFIG}" \
        --base_url "${BASE_URL}"

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
        --model_config "${MODEL_CONFIG}" \
        --base_url "${BASE_URL}"
fi
