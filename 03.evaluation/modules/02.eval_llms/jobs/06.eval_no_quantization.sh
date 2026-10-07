#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "eval_no_quant"
#SBATCH --array=0-1
#SBATCH -o "logs/eval_no_quant_%A_%a.out"
#SBATCH -e "logs/eval_no_quant_%A_%a.err"

mkdir -p logs

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
if [ -z "$SCRIPT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
MODULE_DIR="$(dirname "${SCRIPT_DIR}")"
EVAL_DIR="$(dirname "$(dirname "${MODULE_DIR}")")"

# Fixed Parameters
ONTOLOGY="mp"
EMBED_DIR="kalm"
RAG_CONFIG="${MODULE_DIR}/inputs/best_config.json"
ONTOLOGY_DB="${EVAL_DIR}/../00.data_preparation/toxvaldb/modules/02.prepare_ontologies/outputs/${EMBED_DIR}/${ONTOLOGY}.db"
CONTEXT_FILE="${EVAL_DIR}/../00.data_preparation/toxvaldb/modules/02.prepare_ontologies/inputs/ontology_context/${ONTOLOGY}.txt"
GLOBAL_DEFINITIONS_DB="${EVAL_DIR}/../00.data_preparation/toxvaldb/modules/02.prepare_ontologies/outputs/definitions_${ONTOLOGY}.db"
INPUT_FILE="${EVAL_DIR}/modules/01.eval_experts/outputs/Expanded_average_human.csv"
TARGET_EMBEDDING_DB="${MODULE_DIR}/outputs/tox_effects_embeddings_${EMBED_DIR}.db"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge
module load cuda/12.8.0
module load gcc
GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH

source "${EVAL_DIR}/env-evaluation/bin/activate"
cd "${MODULE_DIR}"

CONSENSUS=1
MODE=2
DEFS="false"
DEFS_ARG=""
DEFS_SUFFIX="no_defs"
SYSTEM_PROMPT="${MODULE_DIR}/config/system_prompt_${MODE}.txt"
CANDIDATE_POOL_FILE="${EVAL_DIR}/../02.expert_annotation/modules/01.create_form/outputs/expert_annotation_${ONTOLOGY}.html"

INDEX="${SLURM_ARRAY_TASK_ID:-0}"

case $INDEX in
    0)
        MODEL="Qwen3.5-35B-A3B"
        PORT="8005"
        NODE="hl-codon-gpu-020"
        ;;
    1)
        MODEL="Qwen3.6-35B-A3B"
        PORT="8012"
        NODE="codon-gpu-001"
        ;;
    *)
        echo "Error: Invalid index: $INDEX"
        exit 1
        ;;
esac

BASE_URL="http://${NODE}.ebi.ac.uk:${PORT}/v1"
MODEL_CONFIG="/nfs/research/ewald/ricos/code/playground/models/config/${MODEL}.json"

OUT_FOLDER="${MODULE_DIR}/outputs/no_quantization"
mkdir -p "${OUT_FOLDER}"

for RUN in {1..10}; do
    OUTPUT_FILE="${OUT_FOLDER}/ToxValLlm_llm_${MODEL}_ontology_${ONTOLOGY}_mode_${MODE}_consensus_${CONSENSUS}_${DEFS_SUFFIX}_run_${RUN}.csv"
    REASONING_FILE="${OUT_FOLDER}/ToxValLlm_llm_${MODEL}_ontology_${ONTOLOGY}_mode_${MODE}_consensus_${CONSENSUS}_${DEFS_SUFFIX}_reasoning_run_${RUN}.json"

    echo "================================================================================"
    echo "Running evaluation:"
    echo "  Model             : ${MODEL}"
    echo "  Ontology          : ${ONTOLOGY}"
    echo "  Mode              : ${MODE}"
    echo "  Consensus Runs    : ${CONSENSUS}"
    echo "  Definitions       : ${DEFS}"
    echo "  Run               : ${RUN}"
    echo "================================================================================"

    python3 "${MODULE_DIR}/scripts/01.eval_llms.py" \
        --input_file "${INPUT_FILE}" \
        --output_file "${OUTPUT_FILE}" \
        --reasoning_file "${REASONING_FILE}" \
        --target_column "Phrase" \
        --model_config "${MODEL_CONFIG}" \
        --base_url "${BASE_URL}" \
        --system_prompt_file "${SYSTEM_PROMPT}" \
        --mode ${MODE} \
        --consensus_runs ${CONSENSUS} \
        ${DEFS_ARG} \
        --context_ontology "$(cat ${CONTEXT_FILE})" \
        --rag_config "${RAG_CONFIG}" \
        --embeddings_model_id "${EMBED_DIR}" \
        --ontology_embedding_db "${ONTOLOGY_DB}" \
        --target_embedding_db "${TARGET_EMBEDDING_DB}" \
        --global_definitions_db "${GLOBAL_DEFINITIONS_DB}" \
        --candidate_pool_file "${CANDIDATE_POOL_FILE}" \
        --descriptions_concurrency 10
done

echo "================================================================================"
echo "Expanding LLM responses..."
echo "================================================================================"
python3 "${MODULE_DIR}/scripts/03.expand_llm_responses.py" \
    --input_dir "${OUT_FOLDER}" \
    --owl_file "${EVAL_DIR}/../00.data_preparation/toxvaldb/modules/02.prepare_ontologies/inputs/owl_files/${ONTOLOGY}.ttl" \
    --output_dir "${OUT_FOLDER}" \
    --ontology "${ONTOLOGY}"

echo "Done."
