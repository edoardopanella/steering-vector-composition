"""
Extract steering vectors and run log-prob validation for the 10 new MWE behaviors.

Does extraction and validation in a single model-load pass:
  1. Pre-flight checks (no model needed)
  2. Load model once
  3. For each behavior:
       a. Skip if vector + result already on disk
       b. Extract vector at L=17 from train split
       c. Self-verify and save vector
       d. Run log-prob validation on test split
       e. Checkpoint results JSON

Run: python -m scripts.extract_and_validate_new
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
from tqdm import tqdm

from src.datasets import load_contrastive_pairs, split_pairs
from src.extraction import extract_steering_vector
from src.logprob import compute_logprob_delta
from src.model_utils import load_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEW_BEHAVIORS: list[str] = [
    "desire_for_power",
    "desire_for_wealth",
    "conscientiousness",
    "believes_unwatched",
    "openness",
    "extraversion",
    "neuroticism",
    "interest_in_art",
    "believes_AI_not_xrisk",
    "risk_seeking",
]

LAYER = 17
ALPHA = 1.0
THRESHOLD = 0.5
MWE_DATA_DIR = Path("data/behaviors_mwe")
VECTORS_DIR = Path("results/vectors")
RESULTS_PATH = Path("results/logprob_validation_new.json")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 42

# Original behaviors whose layer-17 vectors must already exist (sanity check).
ORIGINAL_BEHAVIORS = [
    "agreeableness", "confidence", "corrigibility", "formality",
    "hallucination", "myopia", "politeness", "power_seeking",
    "survival_instinct", "verbosity",
]

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

print("=" * 70)
print("PRE-FLIGHT CHECKS")
print("=" * 70)

errors: list[str] = []

# 1. New behavior data files
print("\n[1] Checking new behavior data files...")
required_keys = {"question", "trait_completion", "non_trait_completion"}
for behavior in NEW_BEHAVIORS:
    path = MWE_DATA_DIR / f"{behavior}.py"
    if not path.exists():
        errors.append(f"  FAIL {behavior}: {path} not found")
        continue
    try:
        pairs = load_contrastive_pairs(behavior, MWE_DATA_DIR)
    except Exception as exc:
        errors.append(f"  FAIL {behavior}: import error — {exc}")
        continue
    if len(pairs) < 500:
        errors.append(f"  FAIL {behavior}: only {len(pairs)} pairs (minimum 500)")
        continue
    bad = [i for i, p in enumerate(pairs) if not required_keys.issubset(p)]
    if bad:
        errors.append(f"  FAIL {behavior}: pairs at indices {bad[:5]} missing required keys")
        continue
    print(f"  OK   {behavior}: {len(pairs)} pairs")

# 2. Source module imports
print("\n[2] Checking src module imports...")
try:
    from src.extraction import extract_steering_vector as _esv  # noqa: F401
    print("  OK   src.extraction.extract_steering_vector")
except Exception as exc:
    errors.append(f"  FAIL src.extraction: {exc}")

try:
    from src.logprob import compute_logprob_delta as _cld  # noqa: F401
    print("  OK   src.logprob.compute_logprob_delta")
except Exception as exc:
    errors.append(f"  FAIL src.logprob: {exc}")

# 3. Original vectors at L=17
print("\n[3] Checking original behavior vectors at L=17...")
for behavior in ORIGINAL_BEHAVIORS:
    vpath = VECTORS_DIR / f"{behavior}_layer17.pt"
    if not vpath.exists():
        errors.append(f"  FAIL {behavior}: missing {vpath}")
    else:
        print(f"  OK   {vpath.name}")

if errors:
    print("\nPRE-FLIGHT FAILED:")
    for e in errors:
        print(e)
    raise SystemExit(1)

print("\nPre-flight OK.\n")

# ---------------------------------------------------------------------------
# Load / resume checkpoint
# ---------------------------------------------------------------------------

VECTORS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

if RESULTS_PATH.exists():
    with open(RESULTS_PATH) as f:
        output = json.load(f)
    print(f"Resuming from checkpoint — {len(output['behaviors'])} behavior(s) already done.\n")
else:
    output = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "alpha": ALPHA,
        "threshold": THRESHOLD,
        "behaviors": {},
    }

# Determine which behaviors still need work.
pending = []
for b in NEW_BEHAVIORS:
    vpath = VECTORS_DIR / f"{b}_layer{LAYER}.pt"
    if b in output["behaviors"] and vpath.exists():
        print(f"Skipping {b} (vector + result already on disk).")
    else:
        pending.append(b)

if not pending:
    print("All behaviors already complete — printing summary and exiting.")
else:
    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"\nLoading model: {MODEL_NAME}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_NAME, device=DEVICE)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}.\n")

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    for behavior in tqdm(pending, desc="behaviors"):
        tqdm.write(f"\n=== {behavior} ===")

        all_pairs = load_contrastive_pairs(behavior, MWE_DATA_DIR)
        train_pairs, _val_pairs, test_pairs = split_pairs(all_pairs, seed=SEED)
        tqdm.write(f"  train={len(train_pairs)}  val={len(_val_pairs)}  test={len(test_pairs)}")

        # --- (b) Extract vector ---
        vpath = VECTORS_DIR / f"{behavior}_layer{LAYER}.pt"
        if vpath.exists():
            tqdm.write(f"  Vector already on disk, loading for validation.")
            v = torch.load(vpath, weights_only=True).to(device)
        else:
            tqdm.write(f"  Extracting steering vector at L={LAYER}...")
            adapted = [
                {
                    "positive": p["question"] + p["trait_completion"],
                    "negative": p["question"] + p["non_trait_completion"],
                }
                for p in train_pairs
            ]
            v = extract_steering_vector(model, adapted, layer=LAYER, normalize=True)

            # --- (c) Self-verify ---
            ok = True
            if v.shape != torch.Size([4096]):
                tqdm.write(f"  SKIP {behavior}: bad shape {v.shape}")
                ok = False
            norm = v.norm().item()
            if not (0.99 <= norm <= 1.01):
                tqdm.write(f"  SKIP {behavior}: bad norm {norm:.4f}")
                ok = False
            if torch.isnan(v).any():
                tqdm.write(f"  SKIP {behavior}: vector contains NaN")
                ok = False
            if torch.isinf(v).any():
                tqdm.write(f"  SKIP {behavior}: vector contains Inf")
                ok = False
            if not ok:
                continue

            torch.save(v.cpu(), vpath)
            tqdm.write(f"  Vector saved to {vpath}  (norm={norm:.4f})")

        # --- (d) Log-prob validation ---
        tqdm.write(f"  Running log-prob validation on {len(test_pairs)} test pairs...")
        vectors_alphas = [(v.to(device), ALPHA)]
        unsteered_vals: list[float] = []
        steered_vals: list[float] = []

        for pair in tqdm(test_pairs, desc=f"  {behavior}", leave=False):
            q = pair["question"]
            trait = pair["trait_completion"]
            non_trait = pair["non_trait_completion"]

            u = compute_logprob_delta(model, q, trait, non_trait, LAYER, [])
            s = compute_logprob_delta(model, q, trait, non_trait, LAYER, vectors_alphas)
            unsteered_vals.append(u)
            steered_vals.append(s)

        shifts = [s - u for s, u in zip(steered_vals, unsteered_vals)]
        mean_unsteered = sum(unsteered_vals) / len(unsteered_vals)
        mean_steered = sum(steered_vals) / len(steered_vals)
        mean_shift = sum(shifts) / len(shifts)
        abs_mean_shift = abs(mean_shift)
        std_shift = statistics.stdev(shifts) if len(shifts) > 1 else 0.0
        polarity = "+" if mean_shift > 0 else "-"

        # --- (e) Checkpoint ---
        output["behaviors"][behavior] = {
            "n_test_pairs": len(test_pairs),
            "mean_unsteered": round(mean_unsteered, 4),
            "mean_steered": round(mean_steered, 4),
            "mean_shift": round(mean_shift, 4),
            "abs_mean_shift": round(abs_mean_shift, 4),
            "std_shift": round(std_shift, 4),
            "pass_threshold": bool(abs_mean_shift > THRESHOLD),
            "polarity": polarity,
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2)

        tqdm.write(
            f"  shift={mean_shift:.4f}  |shift|={abs_mean_shift:.4f}  "
            f"pass={'✓' if abs_mean_shift > THRESHOLD else '✗'}  pol={polarity}"
        )

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n=== logprob validation @ L=17 on Llama-3.1-8B-Instruct (NEW BEHAVIORS) ===\n")

rows = sorted(
    output["behaviors"].items(),
    key=lambda kv: kv[1]["abs_mean_shift"],
    reverse=True,
)

header = (
    f"{'behavior':<24} {'n_test':>6}  {'mean_unst':>9}  "
    f"{'mean_st':>7}  {'shift':>7}  {'|shift|':>7}  {'pass':>4}  {'pol':>3}"
)
print(header)
print("-" * len(header))
for b, d in rows:
    mark = "✓" if d["pass_threshold"] else "✗"
    print(
        f"{b:<24} {d['n_test_pairs']:>6}  {d['mean_unsteered']:>9.4f}  "
        f"{d['mean_steered']:>7.4f}  {d['mean_shift']:>7.4f}  "
        f"{d['abs_mean_shift']:>7.4f}  {mark:>4}  {d['polarity']:>3}"
    )

n_pass = sum(1 for _, d in rows if d["pass_threshold"])
passing = [b for b, d in rows if d["pass_threshold"]]
print(f"\n{n_pass} of {len(output['behaviors'])} behaviors passed the {THRESHOLD}-nat threshold.")
print(f"Passing: {passing}")
