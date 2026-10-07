#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH -J "rag_eval_summary"
#SBATCH -o "logs/eval_summary_%j.out"
#SBATCH -e "logs/eval_summary_%j.err"

# Environmental setup
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

# Ensure logs directory exists (run from jobs/ dir)
mkdir -p logs

# Activate environment
cd ../../../
source env-evaluation/bin/activate

# Go to scripts dir
cd modules/00.eval_rag/scripts

# Step 1: Summarize best configurations across all models
python3 03.summarize_best_configs.py

# Step 2: Generate the comparison plot
python3 04.plot_model_comparison_highlighted.py

# Step 3: Generate the decision impact plots
python3 06.plot_decision_impact.py