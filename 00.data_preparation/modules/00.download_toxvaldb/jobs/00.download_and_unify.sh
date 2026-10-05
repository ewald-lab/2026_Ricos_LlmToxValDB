#!/bin/bash
#SBATCH --job-name=toxval_download_unify
#SBATCH --output=logs/toxval_job_%j.out
#SBATCH --error=logs/toxval_job_%j.err
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2

SCRIPT_DIR=$(dirname "$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)")
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "$(cd "${SCRIPT_DIR}/../../.." && pwd)/env-data-prep/bin/activate"

cd "${MODULE_DIR}/outputs"

echo "--> Downloading data from Zenodo..."
wget -O "ToxValDB_v97_0_Release.zip" "https://zenodo.org/records/17088058/files/ToxValDB%20v97_0%20Release.zip"

echo "--> Unzipping files..."
unzip -o "ToxValDB_v97_0_Release.zip"

echo "--> Running data unification..."
python "${MODULE_DIR}/scripts/00.unify.py" "ToxValDB_v9.7.0_Unified.tsv"

echo "--> Verifying results..."
if [ -f "ToxValDB_v9.7.0_Unified.tsv" ]; then
    echo "SUCCESS! Unified file created in 'outputs' folder."
else
    echo "ERROR: Unified file was not generated."
fi

echo "--> Cleaning up temporary files..."
rm -f "ToxValDB_v97_0_Release.zip"
rm -rf "Data Excel Files" "Data MySQL Dump File"
rm -f "GitHub Input Files.zip" "ToxValDB v9.7.0 Document Catalog.zip" "ToxValDB v9.7.0 Release README.docx"
