"""
Lock in E_i(1,0) and E_i(0,0) baselines at L*=17 for all 7 surviving behaviors.

Protocol: 20 eval prompts x 25 completions x 2 alphas (steered/unsteered) = 1,000
generations per behavior. All GPU generation happens first; all 1,000 judge calls
are fired concurrently via asyncio.gather. Checkpoints after each behavior so a
cluster failure doesn't lose completed work.

Run: python -m scripts.run_baseline_scoring
"""

import asyncio
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from src.datasets import EVAL_PROMPTS
from src.injection import generate_steered_batch
from src.model_utils import load_model
from src.scoring import BEHAVIOR_PROMPTS, make_behavior_judge

# --- config ---
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
JUDGE_MODEL = "gpt-4.1-mini"
VECTORS_DIR = Path("results/vectors/")
OUT_PATH = Path("results/baselines_layer17.json")
BEHAVIORS = [
    "myopia", "corrigibility", "verbosity", "formality",
    "politeness", "confidence", "agreeableness",
]
LAYER = 17
N_COMPLETIONS = 25
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7
N_VALID_WARN = int(0.8 * len(EVAL_PROMPTS) * N_COMPLETIONS)  # 400
# --------------

# --- pre-flight ---
print("Running pre-flight checks...")
for b in BEHAVIORS:
    path = VECTORS_DIR / f"{b}_layer{LAYER}.pt"
    assert path.exists(), f"Missing vector: {path}"
    v = torch.load(path, weights_only=True)
    assert v.shape == torch.Size([4096]), f"{b}: bad shape {v.shape}"
    norm = v.norm().item()
    assert 0.99 <= norm <= 1.01, f"{b}: bad norm {norm:.4f}"
    assert torch.isnan(v).sum() == 0, f"{b}: contains NaN"
    assert torch.isinf(v).sum() == 0, f"{b}: contains Inf"
    assert b in BEHAVIOR_PROMPTS, f"{b}: missing from BEHAVIOR_PROMPTS"
assert len(EVAL_PROMPTS) == 20, f"Expected 20 eval prompts, got {len(EVAL_PROMPTS)}"
print("Pre-flight OK.\n")

# --- load or resume checkpoint ---
if OUT_PATH.exists():
    with open(OUT_PATH) as f:
        output = json.load(f)
    print(f"Resuming from checkpoint — {len(output['behaviors'])} behavior(s) already done.")
else:
    output = {
        "layer": LAYER,
        "n_prompts": len(EVAL_PROMPTS),
        "n_completions_per_prompt": N_COMPLETIONS,
        "behaviors": {},
    }

# --- helpers ---

def compute_stats(scores_flat: list, n_prompts: int, n_per_prompt: int) -> dict:
    valid = [s for s in scores_flat if s is not None]
    n_valid = len(valid)
    mean = sum(valid) / n_valid if valid else 0.0
    std = math.sqrt(sum((s - mean) ** 2 for s in valid) / n_valid) if n_valid > 1 else 0.0
    per_prompt_mean = []
    for i in range(n_prompts):
        chunk = scores_flat[i * n_per_prompt : (i + 1) * n_per_prompt]
        valid_chunk = [s for s in chunk if s is not None]
        per_prompt_mean.append(round(sum(valid_chunk) / len(valid_chunk), 4) if valid_chunk else 0.0)
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "n_valid": n_valid,
        "per_prompt_mean": per_prompt_mean,
    }


# --- main loop ---
print(f"Loading model: {MODEL}")
model = load_model(MODEL, device=DEVICE)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

for behavior in tqdm(BEHAVIORS, desc="behaviors"):
    if behavior in output["behaviors"]:
        print(f"\nSkipping {behavior} (already complete).")
        continue

    print(f"\n=== {behavior} ===")
    vector = torch.load(VECTORS_DIR / f"{behavior}_layer{LAYER}.pt", weights_only=True).to(DEVICE)
    judge = make_behavior_judge(behavior, model=JUDGE_MODEL)

    # Generate all 1,000 completions (GPU) before any scoring.
    steered_completions: list[tuple[int, str]] = []    # (prompt_idx, text)
    unsteered_completions: list[tuple[int, str]] = []

    for alpha, store in [(1.0, steered_completions), (0.0, unsteered_completions)]:
        label = "steered" if alpha == 1.0 else "unsteered"
        for i, prompt in enumerate(tqdm(EVAL_PROMPTS, desc=f"  gen {label}", leave=False)):
            batch = generate_steered_batch(
                model, prompt, LAYER, vector, alpha, n=N_COMPLETIONS,
                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
            )
            for c in batch:
                store.append((i, c))

    # Score all 1,000 concurrently (I/O).
    async def _score_all():
        steered_tasks = [
            judge(question=EVAL_PROMPTS[i], answer=c)
            for i, c in steered_completions
        ]
        unsteered_tasks = [
            judge(question=EVAL_PROMPTS[i], answer=c)
            for i, c in unsteered_completions
        ]
        return await asyncio.gather(*steered_tasks, *unsteered_tasks)

    print(f"  Scoring {len(steered_completions) + len(unsteered_completions)} completions...")
    raw = loop.run_until_complete(_score_all())
    steered_scores = list(raw[:len(steered_completions)])
    unsteered_scores = list(raw[len(steered_completions):])

    output["behaviors"][behavior] = {
        "steered":   compute_stats(steered_scores,   len(EVAL_PROMPTS), N_COMPLETIONS),
        "unsteered": compute_stats(unsteered_scores, len(EVAL_PROMPTS), N_COMPLETIONS),
    }

    # Checkpoint.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Checkpointed to {OUT_PATH}")

loop.close()

# --- n_valid warnings ---
print("\n--- n_valid check ---")
warnings = []
for b, data in output["behaviors"].items():
    for cond in ("steered", "unsteered"):
        n = data[cond]["n_valid"]
        if n < N_VALID_WARN:
            msg = f"WARNING: {b} {cond} n_valid={n} < {N_VALID_WARN}"
            print(msg)
            warnings.append(msg)
if not warnings:
    print("All n_valid >= 400. OK.")

# --- summary table ---
print("\n--- summary (sorted by mean_steered desc) ---")
rows = [
    (b, data["steered"]["mean"], data["unsteered"]["mean"])
    for b, data in output["behaviors"].items()
]
rows.sort(key=lambda r: r[1], reverse=True)
print(f"{'behavior':<20} {'mean_steered':>12} {'mean_unsteered':>14} {'delta':>8}")
print("-" * 58)
for b, ms, mu in rows:
    print(f"{b:<20} {ms:>12.2f} {mu:>14.2f} {ms - mu:>8.2f}")

print("\nDone.")
