"""
Driver: Stage 3 (held-out steered vs baseline validation) for all 15 traits.

Loads the generation model once and runs every trait sequentially. Depends on
run_extract_all.py having already produced the _response_avg_diff.pt vectors.
Per-trait CSV at results/eval_persona_eval/.../{trait}_steer_response_layer16_coef2.0.csv;
existing CSVs are skipped (idempotent), so re-runs only fill the gaps.

Run:
    python -m scripts.extraction.run_steer_eval_all
"""

from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from src.extraction.generation import COHERENCE_PROMPT, _judge_all, generate_batch
from src.inference.hf_model import load_hf_model
from src.extraction.trait_data import load_trait
from src.judge import OpenAiJudge

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Mirrors TRAITS in run_extract_all.py: 7 Anthropic-released + 8 generated.
# Skip-if-exists below makes evil (done in E7.3) a no-op.
TRAITS = [
    # Anthropic-released
    "apathetic",
    "evil",
    "hallucinating",
    "humorous",
    "impolite",
    "optimistic",
    "sycophantic",
    # Project-generated (E7.5)
    "agreeableness",
    "confidence",
    "corrigibility",
    "formality",
    "myopia",
    "power_seeking",
    "refusal",
    "verbosity",
]

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "gpt-4.1-mini"

# Chen et al. 2025, §B.4: layer 16 is the most informative on Llama-3.1-8B-Instruct.
# output_hidden_states[16] = residual stream after block 15 (0-indexed).
HIDDEN_LAYER = 16
HOOK_LAYER_IDX = HIDDEN_LAYER - 1
COEFF = 2.0

N_PER_QUESTION = 5
MAX_NEW_TOKENS = 600
TEMPERATURE = 1.0
BATCH_SIZE = 8
MAX_CONCURRENT_JUDGES = 5

VECTOR_OUTPUT_DIR = Path("results/persona_vectors/Llama-3.1-8B-Instruct")
EVAL_OUTPUT_DIR = Path("results/eval_persona_eval/Llama-3.1-8B-Instruct")
LOGS_DIR = Path("logs")
# ---------------------------------------------------------------------------


def _build_eval_conversations(artifact, n_per_question: int):
    convs, questions_flat = [], []
    for q in artifact.questions:
        for _ in range(n_per_question):
            convs.append([{"role": "user", "content": q}])
            questions_flat.append(q)
    return convs, questions_flat


def _judge_run(judge_model, eval_prompt, questions, answers, max_concurrent):
    trait_judge = OpenAiJudge(judge_model, eval_prompt, eval_type="0_100")
    coh_judge = OpenAiJudge(judge_model, COHERENCE_PROMPT, eval_type="0_100")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        trait_scores = loop.run_until_complete(
            _judge_all(trait_judge, questions, answers, max_concurrent)
        )
        coh_scores = loop.run_until_complete(
            _judge_all(coh_judge, questions, answers, max_concurrent)
        )
    finally:
        loop.close()
    return trait_scores, coh_scores


def _mean(col) -> float:
    valid = [x for x in col if x is not None and pd.notna(x)]
    return sum(valid) / len(valid) if valid else float("nan")


def _run_stage3(trait: str, model, tok, log_path: Path):
    """Run baseline + steered eval for one trait, save CSV, return (bt, bc, st, sc)."""
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = EVAL_OUTPUT_DIR / f"{trait}_steer_response_layer{HIDDEN_LAYER}_coef{COEFF}.csv"

    vec_path = VECTOR_OUTPUT_DIR / f"{trait}_response_avg_diff.pt"
    full_stack = torch.load(vec_path, map_location="cpu")
    vector = full_stack[HIDDEN_LAYER]

    artifact = load_trait(trait, version="eval")
    convs, questions_flat = _build_eval_conversations(artifact, N_PER_QUESTION)

    with log_path.open("w") as fh:
        with redirect_stdout(fh), redirect_stderr(fh):
            print(f"trait={trait}  eval questions={len(artifact.questions)}")
            print(f"using output_hidden_states[{HIDDEN_LAYER}] → hook on block {HOOK_LAYER_IDX}, coeff={COEFF}")
            print(f"total generations per condition: {len(convs)}")

            print("\n=== baseline (no steering) ===")
            _, base_answers = generate_batch(
                model, tok, convs,
                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, batch_size=BATCH_SIZE,
                steering=None,
            )
            base_trait, base_coh = _judge_run(
                JUDGE_MODEL, artifact.eval_prompt, questions_flat, base_answers, MAX_CONCURRENT_JUDGES
            )

            print(f"\n=== steered (coef={COEFF}, layer_idx={HOOK_LAYER_IDX}, response-only) ===")
            _, steer_answers = generate_batch(
                model, tok, convs,
                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, batch_size=BATCH_SIZE,
                steering=(vector, HOOK_LAYER_IDX, COEFF, "response"),
            )
            steer_trait, steer_coh = _judge_run(
                JUDGE_MODEL, artifact.eval_prompt, questions_flat, steer_answers, MAX_CONCURRENT_JUDGES
            )

            df = pd.DataFrame({
                "question": questions_flat,
                "baseline_answer": base_answers,
                "baseline_trait": base_trait,
                "baseline_coherence": base_coh,
                "steered_answer": steer_answers,
                "steered_trait": steer_trait,
                "steered_coherence": steer_coh,
            })
            df.to_csv(out_csv, index=False)
            print(f"\nsaved {out_csv}")

    bt = _mean(base_trait)
    bc = _mean(base_coh)
    st = _mean(steer_trait)
    sc = _mean(steer_coh)
    return bt, bc, st, sc


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} …")
    model, tok = load_hf_model(MODEL_NAME)
    print(f"Model loaded: hidden={model.config.hidden_size}  n_layers={model.config.num_hidden_layers}\n")

    summary_rows = []

    for i, trait in enumerate(TRAITS, 1):
        print(f"\n[{i}/{len(TRAITS)}] {trait}")

        vec_path = VECTOR_OUTPUT_DIR / f"{trait}_response_avg_diff.pt"
        out_csv = EVAL_OUTPUT_DIR / f"{trait}_steer_response_layer{HIDDEN_LAYER}_coef{COEFF}.csv"

        if not vec_path.exists():
            print(f"  WARNING: vector not found at {vec_path} — skipping")
            summary_rows.append({"trait": trait, "status": "MISSING_VEC"})
            continue

        if out_csv.exists():
            print(f"  [skip] output CSV already exists — parsing for summary")
            df = pd.read_csv(out_csv)
            bt = _mean(df["baseline_trait"])
            bc = _mean(df["baseline_coherence"])
            st = _mean(df["steered_trait"])
            sc = _mean(df["steered_coherence"])
        else:
            log_path = LOGS_DIR / f"steer_eval_{trait}.log"
            print(f"  stage 3: steered eval … ", end="", flush=True)
            bt, bc, st, sc = _run_stage3(trait, model, tok, log_path)
            print(f"done  (log: {log_path})")

        dt = st - bt
        dc = sc - bc
        print(
            f"  [{i}/{len(TRAITS)}] {trait}: "
            f"trait {bt:.2f} → {st:.2f} (Δ {dt:+.2f}), "
            f"coh {bc:.2f} → {sc:.2f} (Δ {dc:+.2f})"
        )
        summary_rows.append({
            "trait": trait, "status": "ok",
            "base_trait": bt, "steer_trait": st, "delta_trait": dt,
            "base_coh": bc, "steer_coh": sc, "delta_coh": dc,
        })

    # ---------------------------------------------------------------------------
    # End-of-run summary table
    # ---------------------------------------------------------------------------
    ok_rows = [r for r in summary_rows if r["status"] == "ok"]
    print()
    print(
        f"{'trait':<20} {'base_trait':>10} {'steer_trait':>11} {'Δ_trait':>8} "
        f"{'base_coh':>9} {'steer_coh':>10} {'Δ_coh':>7}"
    )
    print("-" * 80)
    for r in ok_rows:
        print(
            f"{r['trait']:<20} {r['base_trait']:>10.2f} {r['steer_trait']:>11.2f} "
            f"{r['delta_trait']:>+8.2f} {r['base_coh']:>9.2f} {r['steer_coh']:>10.2f} "
            f"{r['delta_coh']:>+7.2f}"
        )

    n_good = sum(1 for r in ok_rows if r["delta_trait"] > 50)
    print(f"\n{n_good} of {len(ok_rows)} traits show Δ_trait > 50 (Anthropic Figure-13-magnitude steering effect).")


if __name__ == "__main__":
    main()
