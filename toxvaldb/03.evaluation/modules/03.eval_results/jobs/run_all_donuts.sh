#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH -J "all_donuts"
#SBATCH -o "logs/out_all_donuts_%j.out"
#SBATCH -e "logs/err_all_donuts_%j.err"

mkdir -p logs

if [ -n "${SLURM_SUBMIT_DIR}" ] && [ -f "${SLURM_SUBMIT_DIR}/toxvaldb/03.evaluation/modules/03.eval_results/jobs/run_all_donuts.sh" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}/toxvaldb"
elif [ -n "${SLURM_SUBMIT_DIR}" ] && [ -f "${SLURM_SUBMIT_DIR}/03.evaluation/modules/03.eval_results/jobs/run_all_donuts.sh" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
fi

OUTPUT_DIR="${REPO_ROOT}/03.evaluation/modules/03.eval_results/outputs"
MP_FIGURES_DIR="${OUTPUT_DIR}/mp"
MODEL="Qwen3.6-35B-A3B-FP8"
MP_INPUT="${REPO_ROOT}/01.text_to_onto/modules/00.tox_effect_to_onto/outputs/ToxValDB_Classified_mp_${MODEL}.csv"
MP_OWL="${REPO_ROOT}/00.data_preparation/modules/02.prepare_ontologies/inputs/owl_files/mp.ttl"
MP_JSON="${REPO_ROOT}/01.text_to_onto/modules/00.tox_effect_to_onto/outputs/ToxValDB_Reasoning_mp_${MODEL}.json"

source "${REPO_ROOT}/03.evaluation/env-evaluation/bin/activate"

mkdir -p "${OUTPUT_DIR}" "${MP_FIGURES_DIR}"

echo "=== Generating combined figure (panel A: donuts, panel B: MP depth) ==="
echo "  Repo root  : ${REPO_ROOT}"
echo "  Output dir : ${OUTPUT_DIR}"

if [ ! -s "${MP_FIGURES_DIR}/year_depth_stats.csv" ]; then
    echo "=== Preparing MP depth stats for panel B ==="
    python "${REPO_ROOT}/03.evaluation/modules/03.eval_results/scripts/eval_results.py" \
        --input_file  "${MP_INPUT}" \
        --json_file   "${MP_JSON}" \
        --owl_file    "${MP_OWL}" \
        --output_dir  "${MP_FIGURES_DIR}" \
        --root_term   "mammalian phenotype" \
        --threshold   500 \
        --depth_only
fi

python "${REPO_ROOT}/03.evaluation/modules/03.eval_results/scripts/gen_all_ontologies_donuts.py" \
    --repo_root  "${REPO_ROOT}" \
    --output_dir "${OUTPUT_DIR}"

echo "Done."
