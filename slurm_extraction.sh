#!/bin/bash
#SBATCH --job-name=vec-extraction
#SBATCH --output=logs/extraction_%j.out
#SBATCH --error=logs/extraction_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu          # ← change to your cluster's GPU partition name

# ── Cluster modules ──────────────────────────────────────────────────────────
# Uncomment / adapt to your cluster:
# module load python/3.11
# module load cuda/12.1
# ─────────────────────────────────────────────────────────────────────────────

# Project root — adjust this path
cd /path/to/steering-vector-composition
mkdir -p logs

source venv/bin/activate

# Load API keys from .env (never committed — create this file on the cluster)
set -a; source .env; set +a

# If the cluster has limited disk space in $HOME, point the HF cache elsewhere:
# export HF_HOME=/scratch/$USER/hf_cache

echo "Starting extraction — $(date)"
python -m scripts.run_extraction
echo "Extraction done — $(date)"
