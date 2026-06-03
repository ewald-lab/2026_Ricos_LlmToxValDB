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
MODELS_KEYS=("qwen" "nemotron" "harrier" "kalm")

SCRIPT_DIR=$(dirname "$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)")
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CODON_NODE="${1:-${CODON_NODE:-}}"
if [ -z "$CODON_NODE" ]; then
    echo "Error: CODON_NODE is required. Pass it as the first argument or set the CODON_NODE env variable."
    echo "Usage: sbatch $0 <codon-node>  (e.g. sbatch $0 codon-gpu-004)"
    exit 1
fi
BASE_URL="http://${CODON_NODE}.ebi.ac.uk:8006/v1"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge && module load cuda/12.8.0 gcc

source "${PREP_ROOT}/env-data-prep/bin/activate"

ONTOLOGY="${ONTOLOGIES[$(( SLURM_ARRAY_TASK_ID / ${#MODELS_KEYS[@]} ))]}"
EMBED_DIR="${MODELS_KEYS[$(( SLURM_ARRAY_TASK_ID % ${#MODELS_KEYS[@]} ))]}"

SYNC_FILE="${REPO_ROOT}/outputs/${ONTOLOGY}_definitions_done.flag"

ROOT_FLAG=""
if [ "$ONTOLOGY" == "hp" ]; then
    ROOT_FLAG="--root_uri http://purl.obolibrary.org/obo/HP_0000118"
elif [ "$ONTOLOGY" == "mp" ]; then
    ROOT_FLAG="--root_uri http://purl.obolibrary.org/obo/MP_0000001"
fi

if [ $(( SLURM_ARRAY_TASK_ID % ${#MODELS_KEYS[@]} )) -eq 0 ]; then
    echo "First task ($EMBED_DIR). Processing definitions..."

    python "${REPO_ROOT}/scripts/01.process_ontology.py" \
        $ROOT_FLAG \
        --ontology_file "${REPO_ROOT}/inputs/owl_files/${ONTOLOGY}.ttl" \
        --ontology_db "${REPO_ROOT}/outputs/${EMBED_DIR}/${ONTOLOGY}.db" \
        --rag_config "${REPO_ROOT}/config/rag/embeddings.json" \
        --embeddings_model_id "${EMBED_DIR}" \
        --global_definitions_db "${REPO_ROOT}/outputs/definitions_${ONTOLOGY}.db" \
        --prompt_template "${REPO_ROOT}/config/tools/create_description_prompt.txt" \
        --evaluate_prompt_template "${REPO_ROOT}/config/tools/evaluate_description_prompt.txt" \
        --definitions_report "${REPO_ROOT}/outputs/${ONTOLOGY}_definitions_report.json" \
        --model_config "/nfs/research/ewald/ricos/code/playground/models/config/Qwen3.6-35B-A3B-FP8.json" \
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

    python "${REPO_ROOT}/scripts/01.process_ontology.py" \
        $ROOT_FLAG \
        --ontology_file "${REPO_ROOT}/inputs/owl_files/${ONTOLOGY}.ttl" \
        --ontology_db "${REPO_ROOT}/outputs/${EMBED_DIR}/${ONTOLOGY}.db" \
        --rag_config "${REPO_ROOT}/config/rag/embeddings.json" \
        --embeddings_model_id "${EMBED_DIR}" \
        --global_definitions_db "${REPO_ROOT}/outputs/definitions_${ONTOLOGY}.db" \
        --prompt_template "${REPO_ROOT}/config/tools/create_description_prompt.txt" \
        --evaluate_prompt_template "${REPO_ROOT}/config/tools/evaluate_description_prompt.txt" \
        --definitions_report "${REPO_ROOT}/outputs/${ONTOLOGY}_definitions_report.json" \
        --model_config "/nfs/research/ewald/ricos/code/playground/models/config/Qwen3.6-35B-A3B-FP8.json" \
        --base_url "${BASE_URL}"
fi
