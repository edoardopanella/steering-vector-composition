#!/bin/bash
#SBATCH --job-name=vec-extraction
#SBATCH --output=/home/3242106/logs/extraction_%j.out
#SBATCH --error=/home/3242106/logs/extraction_%j.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --qos=stud
#SBATCH --account=3242106
#SBATCH --chdir=/home/3242106/steering-vector-composition
#SBATCH --partition=stud

module purge
module load miniconda3
source activate steering-vector-composition-venv

# Load API keys from .env (never committed — create this file on the cluster)
set -a; source .env; set +a

# If the cluster has limited disk space in $HOME, point the HF cache elsewhere:
# export HF_HOME=/scratch/$USER/hf_cache

echo "Starting extraction — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition
python -u scripts/run_extraction.py

conda deactivate
module unload miniconda3
echo "Extraction done — $(date)"

# tail -f extraction_482017.out
