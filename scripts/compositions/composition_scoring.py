"""
LLM judge for evaluating composition of steering vectors.

Joint injection at L=17 on UNIT-NORMALISED vectors:
    h += α * (v_a_unit + v_b_unit)   at block HOOK_LAYER_IDX, positions="response"

For each of the 36 unordered pairs over the 9 validated traits:
    1. Load composition eval JSON (data/composition_eval/{a}__{b}.json):
       held-out questions + two judge eval_prompts (one per trait).
    2. Generate BASELINE (no steering) and JOINT-STEERED responses on those questions.
    3. Score every response with three judges: trait_a, trait_b, coherence.
       Composition score = mean(trait_a, trait_b).
    4. Save per-pair CSVs (baseline + steered) + aggregate JSON. Idempotent.

Run:
    python -m scripts.compositions.composition_scoring
"""

from __future__ import annotations

import asyncio
import json
from contextlib import redirect_stderr, redirect_stdout
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from src.extraction.generation import COHERENCE_PROMPT, _judge_all, generate_batch
from src.inference.hf_model import load_hf_model
from src.judge import OpenAiJudge

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAITS = [
    # Tier S
    "apathetic",
    "evil",
    "hallucinating",
    "humorous",
    "impolite",
    "sycophantic",
    # Tier A
    "power_seeking",
    "confidence",
    "formality",
]

# Same convention as alpha sweep — power_seeking vector points the wrong way in
# logprob space, so we flip its sign before joint injection.
POLARITY_INVERTED: set[str] = {"power_seeking"}

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "gpt-4.1-mini"

HIDDEN_LAYER = 17
HOOK_LAYER_IDX = HIDDEN_LAYER - 1

# Per-trait coefficient applied to each unit vector before summing into the joint
# steering direction. Pulled from the shared α_unit* of alpha_sweep_l17.
COMPOSITION_ALPHA = 4.0

# LLM-judge stage settings — match E7.8 / E9.7 / alpha-sweep.
N_PER_QUESTION = 5
MAX_NEW_TOKENS = 600
TEMPERATURE = 1.0
BATCH_SIZE = 8
MAX_CONCURRENT_JUDGES = 5

VECTOR_OUTPUT_DIR = Path("results/persona_vectors/Llama-3.1-8B-Instruct")
COMPOSITION_DATA_DIR = Path("data/composition_eval")
SCORES_OUTPUT_DIR = Path("results/composition_scoring_l17/Llama-3.1-8B-Instruct")
SUMMARY_OUT_PATH = Path("results/composition_scoring_l17_summary.json")
LOGS_DIR = Path("logs")
# ---------------------------------------------------------------------------


# === IO helpers ============================================================

def _composition_pairs() -> list[tuple[str, str]]:
    """All unordered (a, b) with a < b over TRAITS."""
    return list(combinations(sorted(TRAITS), 2))


def _load_composition_artifact(trait_a: str, trait_b: str) -> dict:
    a, b = sorted([trait_a, trait_b])
    path = COMPOSITION_DATA_DIR / f"{a}__{b}.json"
    return json.loads(path.read_text())


def _load_unit_vector(trait: str) -> torch.Tensor:
    vec_path = VECTOR_OUTPUT_DIR / f"{trait}_response_avg_diff.pt"
    if not vec_path.exists():
        raise FileNotFoundError(f"Vector not found at {vec_path}")
    stack = torch.load(vec_path, map_location="cpu")
    v = stack[HIDDEN_LAYER]
    if trait in POLARITY_INVERTED:
        v = -v
    return v / v.norm()


def _baseline_csv_path(trait_a: str, trait_b: str) -> Path:
    a, b = sorted([trait_a, trait_b])
    return SCORES_OUTPUT_DIR / f"{a}__{b}_baseline.csv"


def _steered_csv_path(trait_a: str, trait_b: str, alpha: float) -> Path:
    a, b = sorted([trait_a, trait_b])
    return SCORES_OUTPUT_DIR / f"{a}__{b}_joint_alpha{alpha}.csv"


# === LLM helpers ===========================================================

def _build_eval_conversations(questions: list[str], n_per_question: int):
    convs, questions_flat = [], []
    for q in questions:
        for _ in range(n_per_question):
            convs.append([{"role": "user", "content": q}])
            questions_flat.append(q)
    return convs, questions_flat


def _judge_run_composition(
    judge_model: str,
    eval_prompt_a: str,
    eval_prompt_b: str,
    questions: list[str],
    answers: list[str],
    max_concurrent: int,
) -> tuple[list, list, list]:
    """Three judges (trait_a, trait_b, coherence), each scored 0-100."""
    judge_a = OpenAiJudge(judge_model, eval_prompt_a, eval_type="0_100")
    judge_b = OpenAiJudge(judge_model, eval_prompt_b, eval_type="0_100")
    judge_coh = OpenAiJudge(judge_model, COHERENCE_PROMPT, eval_type="0_100")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores_a = loop.run_until_complete(_judge_all(judge_a, questions, answers, max_concurrent))
        scores_b = loop.run_until_complete(_judge_all(judge_b, questions, answers, max_concurrent))
        scores_coh = loop.run_until_complete(_judge_all(judge_coh, questions, answers, max_concurrent))
    finally:
        loop.close()
    return scores_a, scores_b, scores_coh


def _mean(col) -> float:
    valid = [x for x in col if x is not None and pd.notna(x)]
    return sum(valid) / len(valid) if valid else float("nan")


def _summarise_df(df: pd.DataFrame) -> dict:
    return {
        "trait_a_mean": _mean(df["trait_a"]),
        "trait_b_mean": _mean(df["trait_b"]),
        "composition_mean": _mean(df["composition"]),
        "coherence_mean": _mean(df["coherence"]),
    }


# === Stage runners =========================================================

def _run_baseline_composition(
    trait_a: str, trait_b: str, artifact: dict, model, tok, log_path: Path,
) -> dict:
    """Generate + judge unsteered baseline for a pair. Idempotent."""
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = _baseline_csv_path(trait_a, trait_b)
    if out_csv.exists():
        return _summarise_df(pd.read_csv(out_csv))

    convs, questions_flat = _build_eval_conversations(artifact["questions"], N_PER_QUESTION)
    with log_path.open("w") as fh:
        with redirect_stdout(fh), redirect_stderr(fh):
            print(f"pair={trait_a}+{trait_b}  baseline  questions={len(questions_flat)}")
            _, answers = generate_batch(
                model, tok, convs,
                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, batch_size=BATCH_SIZE,
                steering=None,
            )
            scores_a, scores_b, scores_coh = _judge_run_composition(
                JUDGE_MODEL,
                artifact["eval_prompt_a"], artifact["eval_prompt_b"],
                questions_flat, answers, MAX_CONCURRENT_JUDGES,
            )
            df = pd.DataFrame({
                "question": questions_flat,
                "answer": answers,
                "trait_a": scores_a,
                "trait_b": scores_b,
                "coherence": scores_coh,
            })
            df["composition"] = df[["trait_a", "trait_b"]].mean(axis=1)
            df.to_csv(out_csv, index=False)
    return _summarise_df(pd.read_csv(out_csv))


def _run_joint_steered_composition(
    trait_a: str, trait_b: str, artifact: dict, alpha: float,
    model, tok, v_a_unit: torch.Tensor, v_b_unit: torch.Tensor, log_path: Path,
) -> dict:
    """Generate + judge joint-steered eval for a pair. Idempotent.

    Injection direction = (v_a_unit + v_b_unit); scalar coefficient = alpha.
    """
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = _steered_csv_path(trait_a, trait_b, alpha)
    if out_csv.exists():
        return _summarise_df(pd.read_csv(out_csv))

    v_joint = v_a_unit + v_b_unit
    convs, questions_flat = _build_eval_conversations(artifact["questions"], N_PER_QUESTION)
    with log_path.open("w") as fh:
        with redirect_stdout(fh), redirect_stderr(fh):
            print(
                f"pair={trait_a}+{trait_b}  joint α={alpha}  "
                f"layer={HIDDEN_LAYER} (hook block {HOOK_LAYER_IDX})  questions={len(questions_flat)}"
            )
            _, answers = generate_batch(
                model, tok, convs,
                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, batch_size=BATCH_SIZE,
                steering=(v_joint, HOOK_LAYER_IDX, alpha, "response"),
            )
            scores_a, scores_b, scores_coh = _judge_run_composition(
                JUDGE_MODEL,
                artifact["eval_prompt_a"], artifact["eval_prompt_b"],
                questions_flat, answers, MAX_CONCURRENT_JUDGES,
            )
            df = pd.DataFrame({
                "question": questions_flat,
                "answer": answers,
                "trait_a": scores_a,
                "trait_b": scores_b,
                "coherence": scores_coh,
            })
            df["composition"] = df[["trait_a", "trait_b"]].mean(axis=1)
            df.to_csv(out_csv, index=False)
    return _summarise_df(pd.read_csv(out_csv))


# === Orchestration =========================================================

def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} ...")
    model, tok = load_hf_model(MODEL_NAME)
    print(f"Model loaded: hidden={model.config.hidden_size}  n_layers={model.config.num_hidden_layers}\n")

    unit_vectors: dict[str, torch.Tensor] = {}
    for t in TRAITS:
        try:
            unit_vectors[t] = _load_unit_vector(t)
            inv = " (inverted)" if t in POLARITY_INVERTED else ""
            print(f"  loaded unit vector for {t}{inv}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    pairs = _composition_pairs()
    print(f"\nEvaluating {len(pairs)} composition pairs at α={COMPOSITION_ALPHA}\n")

    summary_pairs: list[dict] = []

    for i, (a, b) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {a} + {b}")

        if a not in unit_vectors or b not in unit_vectors:
            print(f"  skipping — missing vector")
            summary_pairs.append({"trait_a": a, "trait_b": b, "status": "MISSING_VEC"})
            continue
        try:
            artifact = _load_composition_artifact(a, b)
        except FileNotFoundError as e:
            print(f"  skipping — {e}")
            summary_pairs.append({"trait_a": a, "trait_b": b, "status": "MISSING_ARTIFACT"})
            continue

        v_a, v_b = unit_vectors[a], unit_vectors[b]

        # --- Baseline (no steering) ---
        base_log = LOGS_DIR / f"composition_{a}__{b}_baseline.log"
        print("  baseline … ", end="", flush=True)
        base = _run_baseline_composition(a, b, artifact, model, tok, base_log)
        print(
            f"trait_a={base['trait_a_mean']:.2f}  trait_b={base['trait_b_mean']:.2f}  "
            f"comp={base['composition_mean']:.2f}  coh={base['coherence_mean']:.2f}"
        )

        # --- Joint steered ---
        steer_log = LOGS_DIR / f"composition_{a}__{b}_alpha{COMPOSITION_ALPHA}.log"
        print(f"  joint α={COMPOSITION_ALPHA} … ", end="", flush=True)
        steered = _run_joint_steered_composition(
            a, b, artifact, COMPOSITION_ALPHA, model, tok, v_a, v_b, steer_log,
        )
        delta_a = steered["trait_a_mean"] - base["trait_a_mean"]
        delta_b = steered["trait_b_mean"] - base["trait_b_mean"]
        delta_comp = steered["composition_mean"] - base["composition_mean"]
        delta_coh = steered["coherence_mean"] - base["coherence_mean"]
        print(
            f"trait_a={steered['trait_a_mean']:.2f} (Δ{delta_a:+.2f})  "
            f"trait_b={steered['trait_b_mean']:.2f} (Δ{delta_b:+.2f})  "
            f"comp={steered['composition_mean']:.2f} (Δ{delta_comp:+.2f})  "
            f"coh={steered['coherence_mean']:.2f} (Δ{delta_coh:+.2f})"
        )

        summary_pairs.append({
            "trait_a": a,
            "trait_b": b,
            "status": "ok",
            "alpha": COMPOSITION_ALPHA,
            "baseline": {k: round(v, 2) for k, v in base.items()},
            "steered": {k: round(v, 2) for k, v in steered.items()},
            "delta": {
                "trait_a": round(delta_a, 2),
                "trait_b": round(delta_b, 2),
                "composition": round(delta_comp, 2),
                "coherence": round(delta_coh, 2),
            },
        })

        # Checkpoint after each pair.
        with open(SUMMARY_OUT_PATH, "w") as f:
            json.dump({
                "model": MODEL_NAME,
                "judge_model": JUDGE_MODEL,
                "layer": HIDDEN_LAYER,
                "alpha": COMPOSITION_ALPHA,
                "vector_normalisation": "unit",
                "injection": "h += alpha * (v_a_unit + v_b_unit)",
                "n_per_question": N_PER_QUESTION,
                "pairs": summary_pairs,
            }, f, indent=2)

    # ---------------------------------------------------------------------------
    # End-of-run table
    # ---------------------------------------------------------------------------
    ok_rows = [r for r in summary_pairs if r["status"] == "ok"]
    print()
    header = (
        f"{'trait_a':<14} {'trait_b':<14} "
        f"{'comp_b':>7} {'comp_s':>7} {'Δ_comp':>7} "
        f"{'a_b':>5} {'a_s':>5} {'Δ_a':>6} "
        f"{'b_b':>5} {'b_s':>5} {'Δ_b':>6} "
        f"{'coh_s':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in ok_rows:
        bs, st, dl = r["baseline"], r["steered"], r["delta"]
        print(
            f"{r['trait_a']:<14} {r['trait_b']:<14} "
            f"{bs['composition_mean']:>7.2f} {st['composition_mean']:>7.2f} {dl['composition']:>+7.2f} "
            f"{bs['trait_a_mean']:>5.1f} {st['trait_a_mean']:>5.1f} {dl['trait_a']:>+6.2f} "
            f"{bs['trait_b_mean']:>5.1f} {st['trait_b_mean']:>5.1f} {dl['trait_b']:>+6.2f} "
            f"{st['coherence_mean']:>6.1f}"
        )
    print(f"\nSummary saved to {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
