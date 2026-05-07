#!/bin/bash

#SBATCH --time=0:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -J "analyze_rag_results"
#SBATCH -o "logs/analyze_%j.out"
#SBATCH -e "logs/analyze_%j.err"

# Activate environment
cd ../../../
source env-evaluation/bin/activate

# Go to scripts dir
cd modules/00.eval_rag/scripts

# Run analysis
python3 02.analyze_results.py
