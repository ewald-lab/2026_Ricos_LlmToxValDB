#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -J "expand_expert_responses"
#SBATCH -o "logs/expand_responses_%j.out"
#SBATCH -e "logs/expand_responses_%j.err"

# ─── Configuration ────────────────────────────────────────────────────────────
# Ontology to process (must match the subfolder name in inputs/ and the OWL filename)
ONTOLOGY="mp"

# Root term to exclude from expanded sets (set to "" to disable)
ROOT_TERM="mammalian phenotype"
# ──────────────────────────────────────────────────────────────────────────────

# Ensure logs directory exists (run from jobs/ dir)
mkdir -p logs

# Resolve paths relative to this script's location
# (Handles Slurm script spooling by falling back to SLURM_SUBMIT_DIR)
if [[ "${BASH_SOURCE[0]}" == /var/spool/slurmd/* ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

MODULE_DIR="$(dirname "${SCRIPT_DIR}")"
EVAL_DIR="$(dirname "$(dirname "${MODULE_DIR}")")"

INPUT_DIR="${MODULE_DIR}/inputs/${ONTOLOGY}"
OWL_FILE="${EVAL_DIR}/../00.data_preparation/toxvaldb/modules/02.prepare_ontologies/inputs/owl_files/${ONTOLOGY}.ttl"
OUTPUT_DIR="${MODULE_DIR}/outputs"

# Activate environment
source "${EVAL_DIR}/env-evaluation/bin/activate"

echo "=== Expanding expert responses ==="
echo "  Ontology  : ${ONTOLOGY}"
echo "  OWL file  : ${OWL_FILE}"
echo "  Input dir : ${INPUT_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Manual terms: excluded from expanded All_Terms_Str"

python3 "${MODULE_DIR}/scripts/run_analysis.py" \
    --input_dir  "${INPUT_DIR}" \
    --owl_file   "${OWL_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --ontology   "${ONTOLOGY}" \
    --root_term  "${ROOT_TERM}"

echo "Done. Expanded files written to: ${OUTPUT_DIR}"
