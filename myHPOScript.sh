#!/bin/sh
#SBATCH --account compsci
#SBATCH --partition=ada
#SBATCH --time=50:00:00
#SBATCH --nodes=1 --ntasks=24
#SBATCH --job-name="ASTGCN_HPO_Job"
#SBATCH --mail-user=hmmden001@uct.ac.za
#SBATCH --mail-type=ALL
module load python/miniconda3-py38-usr-A
source activate stgnnEnv
python3 runAGCRN.py
