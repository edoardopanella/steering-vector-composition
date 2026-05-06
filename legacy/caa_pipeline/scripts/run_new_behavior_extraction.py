"""
Extract steering vectors at layer 17 for the three new behaviors:
survival_instinct, corrigibility, power_seeking.

L* = 17 is frozen. This script extracts only that layer so the output
is directly comparable to existing vectors from the original sweep.

Run: python -m scripts.run_new_behavior_extraction
"""

import sys
from pathlib import Path

import torch

from src.datasets import load_contrastive_pairs, split_pairs
from src.extraction import extract_steering_vector
from src.model_utils import load_model

# --- config ---
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
DATA_DIR = Path("data/behaviors/")
OUT_DIR = Path("results/vectors/")
BEHAVIORS = ["corrigibility"]
SINGLE_LAYER = 17
# --------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading model: {MODEL}")
model = load_model(MODEL, device=DEVICE)

for behavior in BEHAVIORS:
    print(f"\n--- {behavior} ---")
    pairs = load_contrastive_pairs(behavior, DATA_DIR)
    train, val, test = split_pairs(pairs)
    print(f"  {len(train)} train / {len(val)} val / {len(test)} test pairs")

    vector = extract_steering_vector(model, train, SINGLE_LAYER)
    torch.save(vector, OUT_DIR / f"{behavior}_layer{SINGLE_LAYER}.pt")
    print(f"  saved {behavior}_layer{SINGLE_LAYER}.pt")

# Self-verify all three outputs.
print("\n--- self-verification ---")
all_ok = True
for behavior in BEHAVIORS:
    path = OUT_DIR / f"{behavior}_layer{SINGLE_LAYER}.pt"
    v = torch.load(path, weights_only=True)
    n_nan = torch.isnan(v).sum().item()
    n_inf = torch.isinf(v).sum().item()
    norm = v.norm().item()
    norm_ok = 0.99 <= norm <= 1.01
    shape_ok = v.shape == torch.Size([4096])
    ok = shape_ok and norm_ok and n_nan == 0 and n_inf == 0
    status = "OK" if ok else "FAIL"
    print(f"  {behavior}: shape={tuple(v.shape)}  norm={norm:.4f}  nan={n_nan}  inf={n_inf}  [{status}]")
    if not ok:
        all_ok = False

if all_ok:
    print("\nVERIFY OK: all three vectors valid.")
else:
    print("\nVERIFY FAIL: see above.")
    sys.exit(1)

print("\nDone.")
