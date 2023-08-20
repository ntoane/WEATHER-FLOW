#!/bin/sh
#SBATCH --account=a100free
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:a100-2g-10gb:1
#SBATCH --time=50:00:00
#SBATCH --job-name="ASTGCNJob"
CUDA_VISIBLE_DEVICES=$(ncvd)
module load python/miniconda3-py38-usr-A
source activate stgnnEnv
python3 runASTGCN.py
