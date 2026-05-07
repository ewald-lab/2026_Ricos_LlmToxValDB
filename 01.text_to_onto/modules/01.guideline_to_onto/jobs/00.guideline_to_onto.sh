#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH -J "endpoint_to_phenotype"
#SBATCH -o "logs/out_endpoint_to_phenotype_%j.out"
#SBATCH -e "logs/err_endpoint_to_phenotype_%j.err"

# --- Configuration ---
JOBS_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
MODULE_ROOT="$(cd "${JOBS_DIR}/.." && pwd)"
TEXT_TO_ONTO_ROOT="$(cd "${MODULE_ROOT}/../.." && pwd)"
PROJECT_ROOT="$(cd "${TEXT_TO_ONTO_ROOT}/.." && pwd)"
DATA_PREP_ROOT="${PROJECT_ROOT}/00.data_preparation"

echo "Job started at: $(date)"
echo "Module Root: ${MODULE_ROOT}"
echo "Data Prep Root: ${DATA_PREP_ROOT}"

EMBED_DIR="${2:-qwen}"
MODEL="${3:-Qwen3.5-122B-A10B-FP8}"
MODE="${4:-1}"
CONSENSU_RUNS="${5:-3}"
BASE_URL="${BASE_URL:-http://codon-gpu-001.ebi.ac.uk:8003/v1}"

ENDPOINTS_FILE="${DATA_PREP_ROOT}/modules/03.prepare_guidelines/outputs/endpoints/normalized_endpoint_list.csv"
OUTPUT_DIR="${MODULE_ROOT}/outputs"
MODEL_CONFIG="${MODULE_ROOT}/config/models/${MODEL}.json"
SYSTEM_PROMPT="${MODULE_ROOT}/config/system_prompt_endpoint_filter.txt"
RAG_CONFIG="${MODULE_ROOT}/config/rag/embeddings.json"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-/nfs/research/ewald/ricos/huggingface_cache}"

module purge
module load cuda/12.8.0
module load gcc

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
export LD_LIBRARY_PATH="${GCC_LIB_PATH}:${LD_LIBRARY_PATH:-}"

source "${DATA_PREP_ROOT}/env-data-prep/bin/activate"
export PYTHONPATH="${MODULE_ROOT}/scripts:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/guidelines_mapping"

# --- Ontology Loop ---
# We loop over all ontologies to generate mappings and then apply them to the DB
for ONTOLOGY in mp hpath aop hp txpo; do
    echo "--- Processing Ontology: ${ONTOLOGY} ---"
    
    CONTEXT_FILE="${DATA_PREP_ROOT}/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt"
    ONTOLOGY_DB="${DATA_PREP_ROOT}/modules/02.prepare_ontologies/outputs/${EMBED_DIR}/${ONTOLOGY}.db"
    DEFINITIONS_DB="${DATA_PREP_ROOT}/modules/02.prepare_ontologies/outputs/definitions_${ONTOLOGY}_gpt-oss-120b.db"

    if [ ! -f "${ONTOLOGY_DB}" ]; then
        ONTOLOGY_DB="${DATA_PREP_ROOT}/modules/02.prepare_ontologies/inputs/ontology_dbs/${EMBED_DIR}/${ONTOLOGY}.db"
        DEFINITIONS_DB="${DATA_PREP_ROOT}/modules/02.prepare_ontologies/inputs/ontology_dbs/definitions_${ONTOLOGY}_gpt-oss-120b.db"
    fi

    python "${MODULE_ROOT}/scripts/00.guideline_to_onto.py" \
        --input_endpoints "${ENDPOINTS_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        --ontology "${ONTOLOGY}" \
        --ontology_embedding_db "${ONTOLOGY_DB}" \
        --global_definitions_db "${DEFINITIONS_DB}" \
        --model_config "${MODEL_CONFIG}" \
        --base_url "${BASE_URL}" \
        --system_prompt_file "${SYSTEM_PROMPT}" \
        --context_ontology "${CONTEXT_FILE}" \
        --rag_config "${RAG_CONFIG}" \
        --embeddings_model_id "${EMBED_DIR}" \
        --mode "${MODE}" \
        --consensus_runs "${CONSENSU_RUNS}"
done
