"""
Exploratory alpha sweep for verbosity and myopia at L*=17.

Tests whether higher steering coefficients (alpha > 1) rescue the dynamic range
observed in the full baseline scoring. Protocol: EVAL_PROMPTS[:3], 3 completions
per prompt, alpha in [0.0, 1.0, 2.0, 3.0, 4.0].

Run: python -m scripts.run_alpha_sweep
"""

import asyncio
import json
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
OUT_PATH = Path("results/alpha_sweep.json")
BEHAVIORS = ["verbosity", "myopia"]
ALPHAS = [0.0, 1.0, 2.0, 3.0, 4.0]
LAYER = 17
N_COMPLETIONS = 3
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7
PROMPTS = EVAL_PROMPTS[:3]
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
print("Pre-flight OK.\n")

print(f"Loading model: {MODEL}")
model = load_model(MODEL, device=DEVICE)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

output = {"layer": LAYER, "behaviors": {}}

for behavior in BEHAVIORS:
    print(f"\n=== {behavior} — generating ===")
    vector = torch.load(VECTORS_DIR / f"{behavior}_layer{LAYER}.pt", weights_only=True).to(DEVICE)
    judge = make_behavior_judge(behavior, model=JUDGE_MODEL)

    # entries[k] = (alpha, prompt_idx, completion_text), ordered by alpha then prompt then sample
    entries: list[tuple[float, int, str]] = []

    for alpha in tqdm(ALPHAS, desc=f"  {behavior} alphas"):
        for i, prompt in enumerate(PROMPTS):
            batch = generate_steered_batch(
                model, prompt, LAYER, vector, alpha, n=N_COMPLETIONS,
                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
            )
            for c in batch:
                entries.append((alpha, i, c))

    # Score all 45 completions concurrently.
    print(f"  Scoring {len(entries)} completions...")

    async def _score_all():
        return await asyncio.gather(*[
            judge(question=PROMPTS[i], answer=c)
            for _, i, c in entries
        ])

    raw = loop.run_until_complete(_score_all())

    # Organise into output schema.
    behavior_data: dict = {}
    idx = 0
    for alpha in ALPHAS:
        key = f"alpha_{alpha}"
        completions_list = []
        scores_list = []
        for i in range(len(PROMPTS)):
            for _ in range(N_COMPLETIONS):
                _, prompt_idx, completion = entries[idx]
                score = raw[idx]
                completions_list.append({
                    "prompt": PROMPTS[prompt_idx],
                    "completion": completion,
                    "score": score,
                })
                scores_list.append(score)
                idx += 1
        valid = [s for s in scores_list if s is not None]
        behavior_data[key] = {
            "mean": round(sum(valid) / len(valid), 2) if valid else 0.0,
            "n_valid": len(valid),
            "completions": completions_list,
        }

    output["behaviors"][behavior] = behavior_data

loop.close()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUT_PATH}")

# --- dose-response table ---
for behavior in BEHAVIORS:
    print(f"\n=== {behavior} ===")
    print(f"{'alpha':<8} {'mean':>6}    sample completion at this alpha (truncated)")
    for alpha in ALPHAS:
        key = f"alpha_{alpha}"
        data = output["behaviors"][behavior][key]
        sample = data["completions"][0]["completion"][:150].replace("\n", " ")
        print(f"{alpha:<8.1f} {data['mean']:>6.2f}    \"{sample}\"")

print("\nDone.")
