#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH -J "test_gen"
#SBATCH -o "logs/out_test_gen_%j.out"
#SBATCH -e "logs/err_test_gen_%j.err"

ONTOLOGY="hpath"
EMBED_DIR="qwen"

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

export PYTHONPATH="${GIT_ROOT}/01.text_to_onto/modules/00.text_to_onto/scripts:$PYTHONPATH"

python "${MODULE_ROOT}/scripts/00.create_test.py" \
    --input_file "${GIT_ROOT}/00.data_preparation/modules/01.prepare_database/outputs/ToxValDB_v9.7.0_Cleaned_02.tsv" \
    --output_file "${MODULE_ROOT}/outputs/test_candidates.txt" \
    --output_json "${MODULE_ROOT}/outputs/test_pool_${ONTOLOGY}.json" \
    --target_column TOXICOLOGICAL_EFFECT_LLM_CLEANED \
    --std_column TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED \
    --rag_config "${GIT_ROOT}/01.text_to_onto/modules/00.text_to_onto/config/rag/embeddings.json" \
    --embeddings_model_id "${EMBED_DIR}" \
    --target_embedding_db "${GIT_ROOT}/00.data_preparation/modules/01.prepare_database/outputs/embeddings_db/tox_effects_${EMBED_DIR}.db" \
    --ontology_embedding_db "${GIT_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/${EMBED_DIR}/${ONTOLOGY}.db" \
    --output_visualization "${MODULE_ROOT}/figures/test_visualization.png" \
    --num_samples 50 \
    --seed 2011 \
    --model_config "${GIT_ROOT}/01.text_to_onto/modules/00.text_to_onto/config/models/Qwen3.5-122B-A10B-FP8.json" \
    --base_url http://codon-gpu-001.ebi.ac.uk:8003/v1 \
    --system_prompt_file "${GIT_ROOT}/01.text_to_onto/modules/00.text_to_onto/config/system_prompt_1.txt" \
    --ontology_definitions \
    --histopathology \
    #--debug \
    #--liver
