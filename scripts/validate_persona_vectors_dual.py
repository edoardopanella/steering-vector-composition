"""
Dual-measurement single-vector validation for the seven persona vectors.

For each trait in TRAITS, at each alpha in ALPHAS:
  Phase A — Generate a free-form response with the steering vector injected
             at LAYER, then score it 0-100 with a GPT-4.1-mini judge.
  Phase B — Run compute_logprob_delta on MWE-format multiple-choice pairs
             to measure the log-prob shift toward the trait completion.

Both measurements are aggregated into a dose-response curve per behavior,
with cross-instrument monotonicity and best-alpha agreement reported.

Data sources:
  - Eval questions: data/persona_artifacts_eval/{trait}.json if present,
    otherwise falls back to data/persona_artifacts/{trait}.json (WARNING:
    these overlap with extraction questions; download the held-out eval set
    from safety-research/persona_vectors data_generation/trait_data_eval/
    when available).
  - MWE pairs:       data/persona_MWE/{trait}.py
  - Vectors:         results/vectors_persona/{trait}_layer16.pt

Run: python -m scripts.validate_persona_vectors_dual
"""

from __future__ import annotations

import json
import os
import statistics
import importlib.util
from pathlib import Path

import torch
from tqdm import tqdm

from src.datasets import load_contrastive_pairs
from src.injection import make_joint_hook
from src.judge import OpenAiJudge
from src.logprob import compute_logprob_delta
from src.model_utils import hook_name, load_model
from src.scoring import score_behavior

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAITS = ["evil", "sycophantic", "hallucinating", "impolite", "apathetic", "humorous", "optimistic"]
LAYER = 16
ALPHAS = [0.0, 1.0, 2.0, 3.0]
N_GEN_EVAL_QUESTIONS = 10
N_MWE_EVAL_QUESTIONS = 10
N_GENERATIONS_PER_CELL = 1
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
SEED = 42
JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
VECTORS_DIR = Path("results/vectors_persona")
GEN_EVAL_DIR = Path("data/persona_artifacts_eval")
GEN_EVAL_FALLBACK_DIR = Path("data/persona_artifacts")
MWE_DIR = Path("data/persona_MWE")
RESULTS_PATH = Path("results/persona_vectors_dual_validation.json")

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

print("=" * 70)
print("PRE-FLIGHT CHECKS")
print("=" * 70)

errors: list[str] = []
gen_eval_sources: dict[str, Path] = {}  # trait -> path to use for Phase A

# 1. Vectors
print("\n[1] Checking persona vectors...")
for trait in TRAITS:
    vpath = VECTORS_DIR / f"{trait}_layer{LAYER}.pt"
    if not vpath.exists():
        errors.append(f"  FAIL {trait}: vector not found at {vpath}")
        continue
    try:
        v = torch.load(vpath, weights_only=True)
    except Exception as exc:
        errors.append(f"  FAIL {trait}: could not load vector — {exc}")
        continue
    if v.shape != torch.Size([4096]):
        errors.append(f"  FAIL {trait}: bad shape {v.shape}")
        continue
    norm = v.norm().item()
    if not (0.99 <= norm <= 1.01):
        errors.append(f"  FAIL {trait}: bad norm {norm:.4f}")
        continue
    if torch.isnan(v).any() or torch.isinf(v).any():
        errors.append(f"  FAIL {trait}: vector contains NaN/Inf")
        continue
    print(f"  OK   {trait}  norm={norm:.4f}")

# 2. Gen-eval question files (with fallback)
print("\n[2] Checking free-gen eval question files...")
for trait in TRAITS:
    primary = GEN_EVAL_DIR / f"{trait}.json"
    fallback = GEN_EVAL_FALLBACK_DIR / f"{trait}.json"
    if primary.exists():
        try:
            with open(primary) as f:
                artifact = json.load(f)
            if not isinstance(artifact.get("questions"), list) or not artifact["questions"]:
                errors.append(f"  FAIL {trait}: {primary} — 'questions' missing or empty")
                continue
            if not isinstance(artifact.get("eval_prompt"), str):
                errors.append(f"  FAIL {trait}: {primary} — 'eval_prompt' missing")
                continue
            print(f"  OK   {trait}  (from {primary.parent.name}/)")
            gen_eval_sources[trait] = primary
        except Exception as exc:
            errors.append(f"  FAIL {trait}: {primary} — {exc}")
    elif fallback.exists():
        try:
            with open(fallback) as f:
                artifact = json.load(f)
            if not isinstance(artifact.get("questions"), list) or not artifact["questions"]:
                errors.append(f"  FAIL {trait}: fallback {fallback} — 'questions' missing or empty")
                continue
            if not isinstance(artifact.get("eval_prompt"), str):
                errors.append(f"  FAIL {trait}: fallback {fallback} — 'eval_prompt' missing")
                continue
            print(f"  WARN {trait}  (falling back to {fallback.parent.name}/ — overlaps with extraction questions)")
            gen_eval_sources[trait] = fallback
        except Exception as exc:
            errors.append(f"  FAIL {trait}: fallback {fallback} — {exc}")
    else:
        errors.append(
            f"  FAIL {trait}: no eval question file found at {primary} or {fallback}.\n"
            f"    → Download from safety-research/persona_vectors data_generation/trait_data_eval/ "
            f"and save to {primary}"
        )

# 3. MWE eval pair files
print("\n[3] Checking MWE eval pair files...")
required_keys = {"question", "trait_completion", "non_trait_completion"}
for trait in TRAITS:
    mwe_path = MWE_DIR / f"{trait}.py"
    if not mwe_path.exists():
        errors.append(
            f"  FAIL {trait}: MWE pairs not found at {mwe_path}.\n"
            f"    → Author ≥10 MWE pairs and save to that path."
        )
        continue
    try:
        pairs = load_contrastive_pairs(trait, MWE_DIR)
    except Exception as exc:
        errors.append(f"  FAIL {trait}: MWE load error — {exc}")
        continue
    if len(pairs) < 1:
        errors.append(f"  FAIL {trait}: MWE file has 0 pairs")
        continue
    bad = [i for i, p in enumerate(pairs) if not required_keys.issubset(p)]
    if bad:
        errors.append(f"  FAIL {trait}: MWE pairs at {bad[:3]} missing required keys")
        continue
    print(f"  OK   {trait}: {len(pairs)} pairs")

# 4. OpenAI API key
print("\n[4] Checking OPENAI_API_KEY...")
if not os.environ.get("OPENAI_API_KEY"):
    errors.append("  FAIL: OPENAI_API_KEY not set in environment")
else:
    print("  OK   OPENAI_API_KEY found")

# 5. Module imports
print("\n[5] Checking module imports...")
for mod_name, sym in [
    ("src.judge", "OpenAiJudge"),
    ("src.scoring", "score_behavior"),
    ("src.logprob", "compute_logprob_delta"),
    ("src.injection", "make_joint_hook"),
    ("src.model_utils", "load_model"),
]:
    try:
        m = __import__(mod_name, fromlist=[sym])
        getattr(m, sym)
        print(f"  OK   {mod_name}.{sym}")
    except Exception as exc:
        errors.append(f"  FAIL {mod_name}: {exc}")

if errors:
    print("\nPRE-FLIGHT FAILED — fix the issues above before running:")
    for e in errors:
        print(e)
    raise SystemExit(1)

print("\nPre-flight OK.\n")

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
if RESULTS_PATH.exists():
    with open(RESULTS_PATH) as f:
        output = json.load(f)
    print(f"Resuming from checkpoint — {len(output['behaviors'])} behavior(s) already done.\n")
else:
    output = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "alphas": ALPHAS,
        "judge_model": JUDGE_MODEL,
        "n_gen_eval_questions": N_GEN_EVAL_QUESTIONS,
        "n_mwe_eval_questions": N_MWE_EVAL_QUESTIONS,
        "n_generations_per_cell": N_GENERATIONS_PER_CELL,
        "behaviors": {},
    }

pending = [t for t in TRAITS if t not in output["behaviors"]]
for t in TRAITS:
    if t not in pending:
        print(f"Skipping {t} (already in checkpoint).")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

if pending:
    print(f"Loading model: {MODEL_NAME}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_NAME, device=DEVICE)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}.\n")

# ---------------------------------------------------------------------------
# Generation helper (steered, chat-formatted prompt, prepend_bos=False)
# ---------------------------------------------------------------------------

def _generate_steered(prompt_str: str, v: torch.Tensor, alpha: float) -> str:
    """Generate from a chat-formatted prompt with the steering vector at LAYER.

    Uses prepend_bos=False because apply_chat_template already emits the BOS
    token (<|begin_of_text|>). Returns only the model's response tokens decoded.
    """
    tokens = model.to_tokens(prompt_str, prepend_bos=False)  # [1, prompt_len]
    prompt_len = tokens.shape[1]
    eos_id = model.tokenizer.eos_token_id
    h_name = hook_name(LAYER)

    if alpha != 0.0:
        hook = make_joint_hook([(v, alpha)])

    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            if alpha != 0.0:
                logits = model.run_with_hooks(
                    tokens, fwd_hooks=[(h_name, hook)], return_type="logits"
                )
            else:
                logits = model(tokens, return_type="logits")
        probs = torch.softmax(logits[0, -1, :] / TEMPERATURE, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == eos_id:
            break

    response_ids = tokens[0, prompt_len:]
    return model.tokenizer.decode(response_ids, skip_special_tokens=True)


def _build_prompt(question: str) -> str:
    """Chat-formatted prompt with empty system message (neutral conditioning)."""
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": question},
    ]
    try:
        return model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )


def _monotonic_3_of_4(means: list[float]) -> bool:
    """True if ≥2 of 3 consecutive alpha transitions are non-decreasing."""
    transitions = [means[i + 1] >= means[i] for i in range(len(means) - 1)]
    return sum(transitions) >= 2


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

torch.manual_seed(SEED)

for trait in tqdm(pending, desc="traits"):
    tqdm.write(f"\n{'=' * 60}")
    tqdm.write(f"=== {trait} ===")
    tqdm.write(f"{'=' * 60}")

    v = torch.load(VECTORS_DIR / f"{trait}_layer{LAYER}.pt", weights_only=True).to(device)

    # Load eval questions and judge template
    with open(gen_eval_sources[trait]) as f:
        gen_artifact = json.load(f)
    eval_questions = gen_artifact["questions"][:N_GEN_EVAL_QUESTIONS]
    judge = OpenAiJudge(model=JUDGE_MODEL, prompt_template=gen_artifact["eval_prompt"], eval_type="0_100")

    # Load MWE pairs
    mwe_pairs = load_contrastive_pairs(trait, MWE_DIR)[:N_MWE_EVAL_QUESTIONS]

    # -----------------------------------------------------------------------
    # Phase A: free-generation + judge scoring
    # -----------------------------------------------------------------------
    tqdm.write(f"\n  Phase A: {len(eval_questions)} questions × {len(ALPHAS)} alphas")

    judge_scores_by_alpha: dict[str, list[float | None]] = {str(a): [] for a in ALPHAS}

    for q_idx, question in enumerate(eval_questions):
        prompt_str = _build_prompt(question)
        tqdm.write(f"    Q{q_idx + 1}/{len(eval_questions)}: {question[:80]}")
        for alpha in ALPHAS:
            responses = [_generate_steered(prompt_str, v, alpha) for _ in range(N_GENERATIONS_PER_CELL)]
            for response in responses:
                score = score_behavior(response, question, judge)
                judge_scores_by_alpha[str(alpha)].append(score)
            valid = [s for s in judge_scores_by_alpha[str(alpha)][-N_GENERATIONS_PER_CELL:] if s is not None]
            tqdm.write(f"      α={alpha}: score={'n/a' if not valid else f'{sum(valid)/len(valid):.1f}'}")

    # Aggregate Phase A
    judge_agg: dict[str, dict] = {}
    judge_means: list[float] = []
    for alpha in ALPHAS:
        raw = judge_scores_by_alpha[str(alpha)]
        valid = [s for s in raw if s is not None]
        mean = sum(valid) / len(valid) if valid else float("nan")
        std = statistics.stdev(valid) if len(valid) > 1 else 0.0
        judge_agg[str(alpha)] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "n_valid": len(valid),
            "n_invalid": len(raw) - len(valid),
            "raw": [round(s, 4) if s is not None else None for s in raw],
        }
        judge_means.append(mean)

    judge_best_alpha = ALPHAS[judge_means.index(max(judge_means))]
    judge_max_mean = max(judge_means)
    judge_mono = _monotonic_3_of_4(judge_means)

    # -----------------------------------------------------------------------
    # Phase B: log-prob delta on MWE pairs
    # -----------------------------------------------------------------------
    tqdm.write(f"\n  Phase B: {len(mwe_pairs)} MWE pairs × {len(ALPHAS)} alphas")

    logprob_deltas_by_alpha: dict[str, list[float]] = {str(a): [] for a in ALPHAS}

    for pair_idx, pair in enumerate(tqdm(mwe_pairs, desc=f"    {trait} MWE", leave=False)):
        for alpha in ALPHAS:
            va = [(v, alpha)] if alpha != 0.0 else []
            delta = compute_logprob_delta(
                model=model,
                question=pair["question"],
                trait_completion=pair["trait_completion"],
                non_trait_completion=pair["non_trait_completion"],
                layer=LAYER,
                vectors_alphas=va,
            )
            logprob_deltas_by_alpha[str(alpha)].append(delta)

    # Aggregate Phase B
    logprob_agg: dict[str, dict] = {}
    logprob_means: list[float] = []
    base_mean = None
    for alpha in ALPHAS:
        raw = logprob_deltas_by_alpha[str(alpha)]
        mean = sum(raw) / len(raw)
        std = statistics.stdev(raw) if len(raw) > 1 else 0.0
        logprob_agg[str(alpha)] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "raw": [round(d, 4) for d in raw],
        }
        logprob_means.append(mean)
        if alpha == 0.0:
            base_mean = mean

    shifts = {
        str(a): round(logprob_means[i] - base_mean, 4)
        for i, a in enumerate(ALPHAS)
        if a != 0.0
    }
    logprob_best_alpha = ALPHAS[logprob_means.index(max(logprob_means))]
    logprob_max_shift = max(shifts.values())
    logprob_mono = _monotonic_3_of_4(logprob_means)

    # -----------------------------------------------------------------------
    # Phase C: assemble and checkpoint
    # -----------------------------------------------------------------------
    output["behaviors"][trait] = {
        "judge": {
            "scores_by_alpha": judge_agg,
            "monotonic_3_of_4": judge_mono,
            "best_alpha": judge_best_alpha,
            "max_mean_score": round(judge_max_mean, 4),
        },
        "logprob": {
            "deltas_by_alpha": logprob_agg,
            "shifts_relative_to_alpha_0": shifts,
            "monotonic_3_of_4": logprob_mono,
            "best_alpha": logprob_best_alpha,
            "max_shift": round(logprob_max_shift, 4),
        },
        "agreement": {
            "best_alpha_judge": judge_best_alpha,
            "best_alpha_logprob": logprob_best_alpha,
            "alpha_agreement": judge_best_alpha == logprob_best_alpha,
            "monotonic_agreement": judge_mono == logprob_mono,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    tqdm.write(f"  Checkpointed {trait}.")

# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

print("\n\n=== persona vector dual validation ===")
print(f"Model: {MODEL_NAME}, layer {LAYER}\n")

# --- Instrument 1: judge ---
h = f"{'trait':<16} {'α=0':>6}  {'α=1':>6}  {'α=2':>6}  {'α=3':>6}  {'mono':>4}  {'best_α':>6}"
print("--- Instrument 1: judge score (0-100) ---")
print(h)
print("-" * len(h))
for trait in TRAITS:
    if trait not in output["behaviors"]:
        continue
    d = output["behaviors"][trait]["judge"]
    means = [d["scores_by_alpha"][str(a)]["mean"] for a in ALPHAS]
    mono = "✓" if d["monotonic_3_of_4"] else "✗"
    print(
        f"{trait:<16} {means[0]:>6.1f}  {means[1]:>6.1f}  {means[2]:>6.1f}  {means[3]:>6.1f}  "
        f"{mono:>4}  {d['best_alpha']:>6.1f}"
    )
n_judge_mono = sum(1 for t in TRAITS if t in output["behaviors"] and output["behaviors"][t]["judge"]["monotonic_3_of_4"])
print(f"\n{n_judge_mono} of {len(output['behaviors'])} traits show monotonic dose-response under judge measurement.")

# --- Instrument 2: log-prob ---
print()
h2 = (f"{'trait':<16} {'α=0':>6}  {'α=1':>6}  {'α=2':>6}  {'α=3':>6}  "
      f"{'mono':>4}  {'best_α':>6}  {'max_shift':>9}")
print("--- Instrument 2: log-prob delta (nats) ---")
print(h2)
print("-" * len(h2))
for trait in TRAITS:
    if trait not in output["behaviors"]:
        continue
    d = output["behaviors"][trait]["logprob"]
    means = [d["deltas_by_alpha"][str(a)]["mean"] for a in ALPHAS]
    mono = "✓" if d["monotonic_3_of_4"] else "✗"
    print(
        f"{trait:<16} {means[0]:>6.2f}  {means[1]:>6.2f}  {means[2]:>6.2f}  {means[3]:>6.2f}  "
        f"{mono:>4}  {d['best_alpha']:>6.1f}  {d['max_shift']:>9.4f}"
    )
n_lp_mono = sum(1 for t in TRAITS if t in output["behaviors"] and output["behaviors"][t]["logprob"]["monotonic_3_of_4"])
print(f"\n{n_lp_mono} of {len(output['behaviors'])} traits show monotonic dose-response under log-prob measurement.")

# --- Cross-instrument agreement ---
print()
h3 = f"{'trait':<16} {'best_α_judge':>12}  {'best_α_lp':>9}  {'α_agree':>7}  {'mono_agree':>10}"
print("--- Cross-instrument agreement ---")
print(h3)
print("-" * len(h3))
for trait in TRAITS:
    if trait not in output["behaviors"]:
        continue
    ag = output["behaviors"][trait]["agreement"]
    aa = "✓" if ag["alpha_agreement"] else "✗"
    ma = "✓" if ag["monotonic_agreement"] else "✗"
    print(
        f"{trait:<16} {ag['best_alpha_judge']:>12.1f}  {ag['best_alpha_logprob']:>9.1f}  "
        f"{aa:>7}  {ma:>10}"
    )
n_alpha_agree = sum(1 for t in TRAITS if t in output["behaviors"] and output["behaviors"][t]["agreement"]["alpha_agreement"])
n_both_mono = sum(1 for t in TRAITS if t in output["behaviors"]
                  and output["behaviors"][t]["judge"]["monotonic_3_of_4"]
                  and output["behaviors"][t]["logprob"]["monotonic_3_of_4"])
print(f"\nAgreement: {n_alpha_agree} of {len(output['behaviors'])} traits have matching best_α across instruments.")
print(f"Both-monotonic: {n_both_mono} of {len(output['behaviors'])} traits show monotonic response under both instruments.")

# --- Best-alpha dict literals ---
print()
judge_best = {t: output["behaviors"][t]["agreement"]["best_alpha_judge"]
              for t in TRAITS if t in output["behaviors"]}
lp_best = {t: output["behaviors"][t]["agreement"]["best_alpha_logprob"]
           for t in TRAITS if t in output["behaviors"]}
print(f"BEST_ALPHA_JUDGE   = {judge_best}")
print(f"BEST_ALPHA_LOGPROB = {lp_best}")
