#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G
#SBATCH -J "process_all_ontologies"
#SBATCH -o "logs/out_all_ontologies_%A_%a.out"
#SBATCH -e "logs/err_all_ontologies_%A_%a.err"
#SBATCH --array=0-4

ONTOLOGIES=("aop" "hp" "hpath" "mp" "txpo")

SCRIPT_DIR=$(dirname "$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)")
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge && module load cuda/12.8.0 gcc

source "${PREP_ROOT}/env-data-prep/bin/activate"

ONTOLOGY="${ONTOLOGIES[$SLURM_ARRAY_TASK_ID]}"
EMBED_DIR="qwen"

# Select root class URI based on ontology
ROOT_FLAG=""
case "$ONTOLOGY" in
    mp)
        ROOT_FLAG="--root_uri http://purl.obolibrary.org/obo/MP_0000001"
        ;;
    hp)
        ROOT_FLAG="--root_uri http://purl.obolibrary.org/obo/HP_0000001"
        ;;
    txpo)
        ROOT_FLAG="--root_uri http://purl.obolibrary.org/obo/TXPO_0000009"
        ;;
    *)
        ROOT_FLAG=""
        ;;
esac

python "${REPO_ROOT}/scripts/01.process_ontology.py" \
    $ROOT_FLAG \
    --ontology_file "${REPO_ROOT}/inputs/owl_files/${ONTOLOGY}.ttl" \
    --ontology_db "${REPO_ROOT}/outputs/${EMBED_DIR}/${ONTOLOGY}.db" \
    --rag_config "${REPO_ROOT}/config/rag/embeddings.json" \
    --embeddings_model_id "${EMBED_DIR}" \
    --global_definitions_db "${REPO_ROOT}/outputs/definitions_${ONTOLOGY}.db"
