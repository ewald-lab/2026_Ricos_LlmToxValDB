#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH -J "rag_reranker_debug"
#SBATCH -o "debug_rag_reranker.out"
#SBATCH -e "debug_rag_reranker.err"

DEBUG_DIR="/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/01.text_to_onto/modules/04.debug"
PROJECT_ROOT="/nfs/research/ewald/ricos/code/2026_Ricos_LlmToxValDB/01.text_to_onto"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/nfs/research/ewald/ricos/huggingface_cache

module purge
module load cuda/12.8.0
module load gcc

GCC_LIB_PATH="/hps/software/spack/opt/spack/linux-rhel9-cascadelake/gcc-11.2.0/gcc-14.2.0-lwfz56aikpunnn56j24sziq4gzrvizji/lib64"
export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH

source "${PROJECT_ROOT}/env-text-to-onto/bin/activate"

cd "${DEBUG_DIR}"

python "${DEBUG_DIR}/debug_rag_reranker.py"
