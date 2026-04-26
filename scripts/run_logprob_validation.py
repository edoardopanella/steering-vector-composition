"""
Phase 1 deliverable: logprob validation on all 10 behaviors at L*=17.

For each behavior, compute mean unsteered / steered log-prob delta across
the 20%-test split of its MWE dataset. Behaviors with abs(mean_shift) > 0.5 nats
proceed to Phase 3.

Run: python -m scripts.run_logprob_validation
"""

import json
import statistics
from pathlib import Path

import torch
from tqdm import tqdm

from src.datasets import load_contrastive_pairs, split_pairs
from src.logprob import compute_logprob_delta
from src.model_utils import load_model

# --- config ---
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 17
ALPHA = 1.0
THRESHOLD = 0.5
VECTORS_DIR = Path("results/vectors")
MWE_DIR = Path("data/behaviors_mwe")
OUT_PATH = Path("results/logprob_validation_instruct.json")

BEHAVIORS = [
    "agreeableness", "confidence", "corrigibility", "formality",
    "hallucination", "myopia", "politeness", "power_seeking",
    "survival_instinct", "verbosity",
]

POLARITY_INVERTED = {"power_seeking", "survival_instinct"}

# --- pre-flight ---
print("Running pre-flight checks...")
errors = []
for b in BEHAVIORS:
    vpath = VECTORS_DIR / f"{b}_layer{LAYER}.pt"
    if not vpath.exists():
        errors.append(f"{b}: missing vector {vpath}")
    else:
        v = torch.load(vpath, weights_only=True)
        if v.shape != torch.Size([4096]):
            errors.append(f"{b}: bad vector shape {v.shape}")
        norm = v.norm().item()
        if not (0.99 <= norm <= 1.01):
            errors.append(f"{b}: bad vector norm {norm:.4f}")
        if torch.isnan(v).any():
            errors.append(f"{b}: vector contains NaN")
        if torch.isinf(v).any():
            errors.append(f"{b}: vector contains Inf")

    dpath = MWE_DIR / f"{b}.py"
    if not dpath.exists():
        errors.append(f"{b}: missing dataset {dpath}")
    else:
        try:
            pairs = load_contrastive_pairs(b, MWE_DIR)
        except Exception as e:
            errors.append(f"{b}: failed to load dataset: {e}")
            continue
        if not pairs:
            errors.append(f"{b}: empty pairs list")
            continue
        required_keys = {"question", "trait_completion", "non_trait_completion"}
        bad = [i for i, p in enumerate(pairs) if not required_keys.issubset(p)]
        if bad:
            errors.append(f"{b}: pairs at indices {bad[:3]} missing required keys")

if errors:
    for e in errors:
        print(f"FAIL: {e}")
    raise SystemExit("Pre-flight failed.")

print("Pre-flight OK.\n")

# --- load or resume checkpoint ---
if OUT_PATH.exists():
    with open(OUT_PATH) as f:
        output = json.load(f)
    print(f"Resuming from checkpoint — {len(output['behaviors'])} behavior(s) already done.")
else:
    output = {
        "model": MODEL,
        "layer": LAYER,
        "alpha": ALPHA,
        "threshold": THRESHOLD,
        "behaviors": {},
    }

# --- load model ---
print(f"Loading model: {MODEL}")
model = load_model(MODEL, device=DEVICE)
device = next(model.parameters()).device
print()

# --- main loop ---
for behavior in tqdm(BEHAVIORS, desc="behaviors"):
    if behavior in output["behaviors"]:
        tqdm.write(f"Skipping {behavior} (already complete).")
        continue

    tqdm.write(f"\n=== {behavior} ===")

    v = torch.load(VECTORS_DIR / f"{behavior}_layer{LAYER}.pt", weights_only=True).to(device)
    vectors_alphas = [(v, ALPHA)]

    all_pairs = load_contrastive_pairs(behavior, MWE_DIR)
    _, _, test_pairs = split_pairs(all_pairs)
    tqdm.write(f"  test pairs: {len(test_pairs)}")

    unsteered_vals = []
    steered_vals = []

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

    output["behaviors"][behavior] = {
        "n_test_pairs": len(test_pairs),
        "mean_unsteered": round(mean_unsteered, 4),
        "mean_steered": round(mean_steered, 4),
        "mean_shift": round(mean_shift, 4),
        "abs_mean_shift": round(abs_mean_shift, 4),
        "std_shift": round(std_shift, 4),
        "pass_threshold": bool(abs_mean_shift > THRESHOLD),
        "polarity_inverted": behavior in POLARITY_INVERTED,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    tqdm.write(
        f"  Checkpointed. shift={mean_shift:.4f}  |shift|={abs_mean_shift:.4f}  "
        f"pass={abs_mean_shift > THRESHOLD}"
    )

# --- summary table ---
print("\n=== logprob validation @ L=17 on Llama-3.1-8B-Instruct ===\n")

rows = sorted(
    output["behaviors"].items(),
    key=lambda kv: kv[1]["abs_mean_shift"],
    reverse=True,
)

header = f"{'behavior':<20} {'n_test':>6}  {'mean_unst':>9}  {'mean_st':>7}  {'shift':>7}  {'|shift|':>7}  {'pass':>4}  {'pol':>3}"
print(header)
print("-" * len(header))
for b, d in rows:
    pol = "-" if d["polarity_inverted"] else "+"
    mark = "✓" if d["pass_threshold"] else "✗"
    print(
        f"{b:<20} {d['n_test_pairs']:>6}  {d['mean_unsteered']:>9.4f}  "
        f"{d['mean_steered']:>7.4f}  {d['mean_shift']:>7.4f}  "
        f"{d['abs_mean_shift']:>7.4f}  {mark:>4}  {pol:>3}"
    )

n_pass = sum(1 for _, d in rows if d["pass_threshold"])
passing = [b for b, d in rows if d["pass_threshold"]]
print(f"\n{n_pass} of {len(BEHAVIORS)} behaviors passed the {THRESHOLD}-nat threshold.")
print(f"Passing: {passing}")
