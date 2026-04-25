"""
Week 1 entrypoint: extract CAA steering vectors for all behaviors across all layers,
then save them to disk. Run layer selection separately once vectors are available.

For local Mac tests: MODEL = "gpt2-xl", DEVICE = "cpu"
For HPC:            MODEL = "meta-llama/Llama-3.1-8B-Instruct", DEVICE = "cuda"
"""

from pathlib import Path
import torch

from src.datasets import load_contrastive_pairs, split_pairs
from src.extraction import extract_all_layers, extract_steering_vector
from src.model_utils import load_model

# --- config --- (cluster ready)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda"

DATA_DIR = Path("data/behaviors/")
OUT_DIR = Path("results/vectors/")
BEHAVIORS = [
    "corrigibility", "power_seeking", "myopia", "verbosity",
    "formality", "politeness", "confidence", "agreeableness",
    "survival_instinct",
]
# Set to an int to extract only one layer (faster for debugging), None to sweep all layers.
SINGLE_LAYER = None
# --------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading model: {MODEL}")
model = load_model(MODEL, device=DEVICE)

for behavior in BEHAVIORS:
    print(f"\n--- {behavior} ---")
    pairs = load_contrastive_pairs(behavior, DATA_DIR)
    train, val, test = split_pairs(pairs)
    print(f"  {len(train)} train / {len(val)} val / {len(test)} test pairs")

    if SINGLE_LAYER is not None:
        vector = extract_steering_vector(model, train, SINGLE_LAYER)
        torch.save(vector, OUT_DIR / f"{behavior}_layer{SINGLE_LAYER}.pt")
    else:
        layer_vectors = extract_all_layers(model, train)
        for layer, vector in layer_vectors.items():
            torch.save(vector, OUT_DIR / f"{behavior}_layer{layer}.pt")

# Layer selection runs separately once vectors are on disk:
#   python -m scripts.run_layer_selection
print("\nDone.")
