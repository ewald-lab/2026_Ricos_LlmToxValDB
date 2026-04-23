#!/bin/bash
#SBATCH --job-name=toxval_download_unify
#SBATCH --output=logs/toxval_job_%j.out
#SBATCH --error=logs/toxval_job_%j.err
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=ricos@ebi.ac.uk
#SBATCH --mail-type=END

# --- 1. Activate environment ---
SCRIPT_PATH=$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | head -n 1)
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${PREP_ROOT}/env-data-prep/bin/activate"

# --- 2. Move to outputs directory ---
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd $MODULE_DIR/outputs

# --- 3. Download data from Zenodo ---
echo "--> Downloading data from Zenodo..."
wget -O "ToxValDB_v97_0_Release.zip" "https://zenodo.org/records/17088058/files/ToxValDB%20v97_0%20Release.zip"

# --- 4. Unzip files ---
echo "--> Unzipping files..."
unzip -o "ToxValDB_v97_0_Release.zip"

# --- 5. Unify Excel files ---
echo "--> Running data unification..."
# The unify.py script now takes the output path as an argument
# We are currently in 'outputs/', so we save it here
python ../scripts/00.unify.py "ToxValDB_v9.7.0_Unified.tsv"

# --- 6. Verification ---
echo "--> Verifying results..."
if [ -f "ToxValDB_v9.7.0_Unified.tsv" ]; then
    echo "SUCCESS! Unified file created in 'outputs' folder."
else
    echo "ERROR: Unified file was not generated."
fi

# --- 7. Clean up temporary files ---
echo "--> Cleaning up temporary files..."
rm -f "ToxValDB_v97_0_Release.zip"
rm -rf "Data Excel Files" "Data MySQL Dump File"
rm -f "GitHub Input Files.zip" "ToxValDB v9.7.0 Document Catalog.zip" "ToxValDB v9.7.0 Release README.docx"
