#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH -J "test_gen"
#SBATCH -o "logs/out_test_gen_%j.out"
#SBATCH -e "logs/err_test_gen_%j.err"

EMBED_DIR="kalm"

MODEL_CONFIG="/nfs/research/ewald/ricos/code/playground/models/config/${MODEL}.json"
CODON_NODE="${1:-${CODON_NODE:-}}"
if [ -z "$CODON_NODE" ]; then
    echo "Error: CODON_NODE is required. Pass it as the first argument or set the CODON_NODE env variable."
    echo "Usage: sbatch $0 <codon-node>  (e.g. sbatch $0 codon-gpu-001)"
    exit 1
fi
BASE_URL="http://${CODON_NODE}.ebi.ac.uk:8003/v1"

JOBS_DIR="$SLURM_SUBMIT_DIR"
MODULE_ROOT="$(cd "${JOBS_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"
GIT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge
module load cuda/12.8.0
module load gcc

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH

source "${PROJECT_ROOT}/env-expert-annot/bin/activate"


for ONTOLOGY in hpath mp; do
    echo "--- Processing ontology: ${ONTOLOGY} ---"

    EXTRA_FLAGS=""
    if [ "${ONTOLOGY}" = "hpath" ]; then
        EXTRA_FLAGS="--histopathology"
    fi

    python "${MODULE_ROOT}/scripts/00.create_test.py" \
        --input_file "${GIT_ROOT}/00.data_preparation/modules/01.prepare_database/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv" \
        --output_file "${MODULE_ROOT}/outputs/test_candidates_${ONTOLOGY}.txt" \
        --output_json "${MODULE_ROOT}/outputs/test_pool_${ONTOLOGY}.json" \
        --target_column TOXICOLOGICAL_EFFECT_LLM_CLEANED \
        --std_column TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED \
        --rag_config "${MODULE_ROOT}/config/best_config.json" \
        --embeddings_model_id "${EMBED_DIR}" \
        --target_embedding_db "${GIT_ROOT}/00.data_preparation/modules/01.prepare_database/outputs/embeddings_db/tox_effects_${EMBED_DIR}.db" \
        --ontology_embedding_db "${GIT_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/${EMBED_DIR}/${ONTOLOGY}.db" \
        --output_visualization "${MODULE_ROOT}/figures/test_visualization_${ONTOLOGY}.png" \
        --num_samples 50 \
        --seed 2011 \
        --model_config "${MODEL_CONFIG}" \
        --base_url "${BASE_URL}" \
        --system_prompt_file "${GIT_ROOT}/01.text_to_onto/modules/00.text_to_onto/config/system_prompt_1.txt" \
        ${EXTRA_FLAGS}

done
