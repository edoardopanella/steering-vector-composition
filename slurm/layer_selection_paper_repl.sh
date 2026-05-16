#!/bin/bash
#SBATCH --job-name=layer-selection-paper-repl
#SBATCH --output=/home/3247897/logs/layer_selection_paper_repl_%j.out
#SBATCH --error=/home/3247897/logs/layer_selection_paper_repl_%j.err
#SBATCH --time=23:59:00            # 3 traits × (1 baseline + 32 layers) × 100 gens; ~8h observed (job 488481)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G                 # bf16 Llama-3.1-8B + HF generate; matches the other layer_selection jobs
#SBATCH --gres=gpu:1
#SBATCH --qos=stud
#SBATCH --account=3247897
#SBATCH --partition=stud
#SBATCH --chdir=/home/3247897/steering-vector-composition
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=riccardo.scibetta7@gmail.com

set -euo pipefail

export SCRATCH=/mnt/beegfsstudents/home/$USER
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$SCRATCH/torch_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module purge
module load miniconda3
source activate steering-vector-composition-venv

# OPENAI_API_KEY (and anything else) lives in .env at repo root.
set -a; source .env; set +a

mkdir -p /home/3247897/logs

echo "Starting layer-selection paper-repl — $(date)"
export PYTHONPATH=/home/3247897/steering-vector-composition
python -u scripts/layer_selection/run_paper_repl.py
echo "layer-selection paper-repl done — $(date)"
