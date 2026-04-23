#!/bin/bash
#SBATCH --job-name=layer-selection
#SBATCH --output=logs/layer_selection_%j.out
#SBATCH --error=logs/layer_selection_%j.err
#SBATCH --time=12:00:00          # generation (GPU) + ~6 h of judge API calls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu          # ← change to your cluster's GPU partition name

# ── Cluster modules ──────────────────────────────────────────────────────────
# module load python/3.11
# module load cuda/12.1
# ─────────────────────────────────────────────────────────────────────────────

# !! IMPORTANT: this script makes outbound HTTPS calls to the OpenAI API.
# Check whether your cluster's compute nodes have internet access.
# If not, you need to run this from a login node or a gateway node instead.

cd /path/to/steering-vector-composition
mkdir -p logs

source venv/bin/activate

# Load API keys from .env (never committed — create this file on the cluster)
set -a; source .env; set +a

echo "Starting layer selection — $(date)"
python -m scripts.run_layer_selection
echo "Layer selection done — $(date)"
