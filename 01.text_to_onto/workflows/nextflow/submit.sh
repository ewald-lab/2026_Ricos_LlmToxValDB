#!/bin/bash
#SBATCH --job-name=nf-pipeline
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00

# 1. Define fixed absolute paths
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NF_DIR="${REPO_ROOT}/analysis/nextflow"

module load nextflow 2>/dev/null || true

# 2. Enter module root
cd "${REPO_ROOT}"

# 3. Merge parameter files
MERGED_PARAMS=$(mktemp /tmp/nf_params_XXXXX.json)
python3 -c "
import json
merged = {}
files = [
    '${NF_DIR}/params/params_ontology.json',
    '${NF_DIR}/params/params_mapping.json'
]
for f in files:
    with open(f) as fh:
        merged.update(json.load(fh))
json.dump(merged, open('${MERGED_PARAMS}', 'w'), indent=4)
"

echo "Pipeline starting at: $(date)"
echo "Repo Root: ${REPO_ROOT}"
echo "Config Dir: ${NF_DIR}"

# 4. Run Nextflow
nextflow run "${NF_DIR}/main.nf" \
    -params-file "${MERGED_PARAMS}" \
    -resume \
    "$@"

EXIT_CODE=$?
rm -f "${MERGED_PARAMS}"

# 5. Cleanup: remove Nextflow logs
rm -f "${REPO_ROOT}"/.nextflow.log*

echo "Pipeline finished at: $(date) with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
