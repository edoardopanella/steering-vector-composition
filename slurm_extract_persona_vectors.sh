#!/bin/bash
#SBATCH --job-name=extract-persona-vectors
#SBATCH --output=/home/3242106/logs/extract_persona_vectors_%j.out
#SBATCH --error=/home/3242106/logs/extract_persona_vectors_%j.err
#SBATCH --time=08:00:00
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

echo "Starting extract-persona-vectors — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned
python -u -m scripts.extract_persona_vectors
echo "Extract-persona-vectors done — $(date)"
