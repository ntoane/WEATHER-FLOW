#!/bin/sh
#SBATCH --account compsci
#SBATCH --partition=ada
#SBATCH --time=20:00:00
#SBATCH --nodes=1 --ntasks=8
#SBATCH --job-name="6h_ASTGCN_Job"
#SBATCH --mail-user=hmmden001@uct.ac.za
#SBATCH --mail-type=ALL
module load python/miniconda3-py38-usr-A
source activate stgnnEnv
python3 runASTGCN_6h.py
