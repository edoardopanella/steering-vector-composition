"""
Extract persona vectors for seven trait behaviors following Chen et al. (2025).

Loads Llama-3.1-8B-Instruct once, then iterates over all seven traits.
For each trait, calls extract_persona_vector (src/extraction.py) which:
  - Builds chat-formatted prompts from (priming pair, question) combinations
  - Generates responses unsteered (temperature=1.0, max_new_tokens=512)
  - Re-runs each full sequence through run_with_cache at layer 16
  - Mean-pools response-token activations → mean(pos) - mean(neg)

v1 settings: 1 rollout per combo, no judge filtering.
5 pairs × 20 questions × 2 conditions × 1 rollout = 200 generations/trait.

Run: python -m scripts.extract_persona_vectors
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.extraction import extract_persona_vector
from src.model_utils import load_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAITS = ["evil", "sycophantic", "hallucinating", "impolite", "apathetic", "humorous", "optimistic"]
LAYER = 16
ARTIFACT_DIR = Path("data/persona_artifacts")
VECTORS_DIR = Path("results/vectors_persona")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
N_ROLLOUTS = 1
SEED = 42

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

print("=" * 60)
print("PRE-FLIGHT CHECKS")
print("=" * 60)

errors: list[str] = []
artifacts: dict[str, dict] = {}

for trait in TRAITS:
    path = ARTIFACT_DIR / f"{trait}.json"
    if not path.exists():
        errors.append(f"  FAIL {trait}: {path} not found")
        continue
    try:
        with open(path) as f:
            artifact = json.load(f)
    except Exception as exc:
        errors.append(f"  FAIL {trait}: JSON parse error — {exc}")
        continue

    instructions = artifact.get("instruction")
    questions = artifact.get("questions")
    if not isinstance(instructions, list) or not instructions:
        errors.append(f"  FAIL {trait}: 'instruction' missing or empty")
        continue
    bad_pairs = [i for i, p in enumerate(instructions)
                 if not isinstance(p.get("pos"), str) or not isinstance(p.get("neg"), str)]
    if bad_pairs:
        errors.append(f"  FAIL {trait}: instruction pairs at {bad_pairs} missing pos/neg keys")
        continue
    if not isinstance(questions, list) or not questions:
        errors.append(f"  FAIL {trait}: 'questions' missing or empty")
        continue

    n_combos = len(instructions) * len(questions) * 2 * N_ROLLOUTS
    print(f"  OK   {trait}: {len(instructions)} pairs, {len(questions)} questions → {n_combos} generations")
    artifacts[trait] = artifact

if errors:
    print("\nPRE-FLIGHT FAILED:")
    for e in errors:
        print(e)
    raise SystemExit(1)

print("\nPre-flight OK.\n")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

VECTORS_DIR.mkdir(parents=True, exist_ok=True)

pending = []
for trait in TRAITS:
    vpath = VECTORS_DIR / f"{trait}_layer{LAYER}.pt"
    if vpath.exists():
        print(f"Skipping {trait} (vector already on disk).")
    else:
        pending.append(trait)

if not pending:
    print("All traits already complete.")
else:
    print(f"\nLoading model: {MODEL_NAME}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_NAME, device=DEVICE)
    print(f"Model loaded on {next(model.parameters()).device}.\n")

# ---------------------------------------------------------------------------
# Extraction loop
# ---------------------------------------------------------------------------

results: list[tuple[str, str, int, float]] = []  # (trait, status, n_gen, norm)

for trait in TRAITS:
    vpath = VECTORS_DIR / f"{trait}_layer{LAYER}.pt"

    if vpath.exists():
        v = torch.load(vpath, weights_only=True)
        results.append((trait, "skipped", 0, v.norm().item()))
        continue

    print(f"\n=== {trait} ===")
    artifact = artifacts[trait]
    n_gen = len(artifact["instruction"]) * len(artifact["questions"]) * 2 * N_ROLLOUTS

    try:
        v = extract_persona_vector(
            model,
            artifact,
            layer=LAYER,
            n_rollouts_per_combo=N_ROLLOUTS,
            max_new_tokens=512,
            temperature=1.0,
            normalize=True,
            seed=SEED,
        )
    except Exception as exc:
        print(f"  ERROR during extraction: {exc}")
        results.append((trait, f"error: {exc}", n_gen, float("nan")))
        continue

    # Self-verify
    ok = True
    if v.shape != torch.Size([4096]):
        print(f"  SKIP {trait}: bad shape {v.shape}")
        ok = False
    norm = v.norm().item()
    if not (0.99 <= norm <= 1.01):
        print(f"  SKIP {trait}: bad norm {norm:.4f}")
        ok = False
    if torch.isnan(v).any():
        print(f"  SKIP {trait}: vector contains NaN")
        ok = False
    if torch.isinf(v).any():
        print(f"  SKIP {trait}: vector contains Inf")
        ok = False

    if not ok:
        results.append((trait, "failed self-verify", n_gen, norm))
        continue

    torch.save(v.cpu(), vpath)
    print(f"  {trait}: vector saved  norm={norm:.4f}")
    results.append((trait, "ok", n_gen, norm))

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"{'trait':<16} {'status':<20} {'n_gen':>6}  {'vector_norm':>11}")
print("-" * 60)
for trait, status, n_gen, norm in results:
    norm_str = f"{norm:.4f}" if not (norm != norm) else "nan"  # nan check
    print(f"{trait:<16} {status:<20} {n_gen:>6}  {norm_str:>11}")

n_ok = sum(1 for _, s, _, _ in results if s == "ok" or s == "skipped")
print(f"\n{n_ok}/{len(TRAITS)} traits have vectors on disk.")
