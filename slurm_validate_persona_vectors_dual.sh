#!/bin/bash
#SBATCH --job-name=persona-dual-validation
#SBATCH --output=/home/3242106/logs/persona_dual_validation_%j.out
#SBATCH --error=/home/3242106/logs/persona_dual_validation_%j.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --partition=stud
#SBATCH --account=3242106
#SBATCH --chdir=/home/3242106/steering-vector-composition-cloned

module purge
module load miniconda3
source activate steering-vector-composition-venv

set -a; source .env; set +a

echo "Starting persona dual validation — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned
python -u -m scripts.validate_persona_vectors_dual
echo "Persona dual validation done — $(date)"
