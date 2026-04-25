#!/bin/bash
#SBATCH --job-name=new-behavior-validation
#SBATCH --output=/home/3242106/logs/new_behavior_validation_%j.out
#SBATCH --error=/home/3242106/logs/new_behavior_validation_%j.err
#SBATCH --time=02:00:00          # 3 behaviors x 3 prompts x 2 alphas + judge API calls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --partition=stud
#SBATCH --account=3242106
#SBATCH --chdir=/home/3242106/steering-vector-composition-cloned

module purge
module load miniconda3
source activate steering-vector-composition-venv

set -a; source .env; set +a

echo "Starting new behavior validation — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned
python -u scripts/run_new_behavior_validation.py
echo "New behavior validation done — $(date)"
