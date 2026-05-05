#!/bin/bash
#SBATCH --job-name=new-behavior-extraction
#SBATCH --output=/home/3242106/logs/new_behavior_extraction_%j.out
#SBATCH --error=/home/3242106/logs/new_behavior_extraction_%j.err
#SBATCH --time=02:00:00          # 3 behaviors x 1 layer; faster than full sweep
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

echo "Starting new behavior extraction — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned

echo "myopia vectors before: $(ls results/vectors/myopia_layer*.pt 2>/dev/null | wc -l)"

python -u scripts/run_new_behavior_extraction.py

echo "myopia vectors after: $(ls results/vectors/myopia_layer*.pt 2>/dev/null | wc -l)"
echo "New behavior extraction done — $(date)"
