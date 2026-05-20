#!/bin/bash

#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH -J "toxval_llm_production_run"
#SBATCH -o "logs/out_tvllm_prod_%j.out"
#SBATCH -e "logs/err_tvllm_prod_%j.err"

MODE=1
ONTOLOGY="mp"
CONSENSU_RUNS=3
MODEL="Qwen3.5-122B-A10B-FP8"
NUM_BACKUPS=10
EMBED_DIR="qwen"

JOBS_DIR="$SLURM_SUBMIT_DIR"
REPO_ROOT="$(cd "${JOBS_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"
GIT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
RAG_CONFIG="${REPO_ROOT}/config/rag/best_config.json"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge
module load cuda/12.8.0
module load gcc

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH

source "${PROJECT_ROOT}/env-text-to-onto/bin/activate"

export PYTHONPATH="${REPO_ROOT}/scripts:$PYTHONPATH"

python "${REPO_ROOT}/scripts/00.tox_effect_to_onto.py" \
    --input_file "${GIT_ROOT}/00.data_preparation/modules/01.prepare_database/outputs/ToxValDB_v9.7.0_Cleaned_03.tsv" \
    --output_file "${REPO_ROOT}/outputs/ToxValDB_Classified_${ONTOLOGY}_${MODEL}.csv" \
    --reasoning_file "${REPO_ROOT}/outputs/ToxValDB_Reasoning_${ONTOLOGY}_${MODEL}.json" \
    --target_column TOXICOLOGICAL_EFFECT_LLM_CLEANED \
    --std_column TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED \
    --system_prompt_file "${REPO_ROOT}/config/system_prompt_${MODE}.txt" \
    --rag_config ${RAG_CONFIG} \
    --embeddings_model_id ${EMBED_DIR} \
    --ontology_embedding_db "${GIT_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/${EMBED_DIR}/${ONTOLOGY}.db" \
    --global_definitions_db "${GIT_ROOT}/00.data_preparation/modules/02.prepare_ontologies/outputs/definitions_${ONTOLOGY}_gpt-oss-120b.db" \
    --target_embedding_db "${GIT_ROOT}/00.data_preparation/modules/01.prepare_database/outputs/embeddings_db/tox_effects_${EMBED_DIR}.db" \
    --mode ${MODE} \
    --model_config "${REPO_ROOT}/config/models/${MODEL}.json" \
    --base_url http://codon-gpu-001.ebi.ac.uk:8003/v1 \
    --consensus_runs ${CONSENSU_RUNS} \
    --ontology_definitions \
    --num_backups ${NUM_BACKUPS} \
    --context_ontology "$(cat "${GIT_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt")"
