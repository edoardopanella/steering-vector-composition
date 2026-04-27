"""
Stage 3 of the Anthropic persona-vectors replication: steered-eval sanity check.

Loads response_avg_diff.pt[layer], generates on the held-out trait_data_eval set
both with and without steering, and judges both with the trait + coherence judges.
Acceptance: trait score under steering should be substantially higher than baseline,
mirroring Figure 2 of the paper.

Run (defaults to evil for backward compat):
    python -m scripts.anthropic_repl.run_steer_eval
    python -m scripts.anthropic_repl.run_steer_eval --trait sycophantic --coef 1.5
    python -m scripts.anthropic_repl.run_steer_eval --trait hallucinating

Pre-flight checks: vector .pt exists for the trait, eval JSON exists in
anthropic_code/data_generation/trait_data_eval/{trait}.json.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from src.anthropic_repl.generation import (
    COHERENCE_PROMPT,
    _judge_all,
    generate_batch,
)
from src.anthropic_repl.hf_model import load_hf_model
from src.anthropic_repl.trait_data import TRAIT_DATA_DIR, load_trait
from src.judge import OpenAiJudge

load_dotenv()

# --- config ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "gpt-4.1-mini"

# Chen et al. 2025, §B.4: "layer 16 is the most informative for all three traits"
# on Llama-3.1-8B-Instruct. Paper uses 1-based layer indexing referring to the
# *output* of that layer. So "layer 16 activation" = output_hidden_states[16]
# = forward hook on model.model.layers[15] (0-indexed).
DEFAULT_HIDDEN_LAYER = 16
DEFAULT_COEFF = 2.0

N_PER_QUESTION = 5
MAX_NEW_TOKENS = 600
TEMPERATURE = 1.0
BATCH_SIZE = 8
# Lowered from 50 to stay within OpenAI TPM (200K/min) + RPM (500/min) limits.
MAX_CONCURRENT_JUDGES = 5

VECTORS_DIR = Path("results/anthropic_repl/persona_vectors") / MODEL_NAME.split("/")[-1]
OUT_DIR = Path("results/anthropic_repl/eval_persona_eval") / MODEL_NAME.split("/")[-1]
# --------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="evil")
    p.add_argument("--hidden-layer", type=int, default=DEFAULT_HIDDEN_LAYER, help="output_hidden_states index — paper's 1-indexed 'layer N'")
    p.add_argument("--coef", type=float, default=DEFAULT_COEFF)
    return p.parse_args()


def preflight(trait: str, hidden_layer: int) -> Path:
    vec_path = VECTORS_DIR / f"{trait}_response_avg_diff.pt"
    eval_json = TRAIT_DATA_DIR / "trait_data_eval" / f"{trait}.json"
    missing = [str(p) for p in (vec_path, eval_json) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files (run earlier stages first or check trait name):\n  "
            + "\n  ".join(missing)
        )
    # Spot-check vector shape: (n_layers+1, hidden_dim).
    stack = torch.load(vec_path, map_location="cpu")
    if stack.ndim != 2 or stack.shape[1] != 4096:
        raise RuntimeError(f"Vector at {vec_path} has unexpected shape {tuple(stack.shape)}; expected (33, 4096)")
    if not (0 <= hidden_layer < stack.shape[0]):
        raise IndexError(f"hidden_layer={hidden_layer} out of range for vector stack of length {stack.shape[0]}")
    return vec_path


def _build_eval_conversations(artifact, n_per_question):
    """Plain user prompts only — eval set tests whether the vector itself elicits
    the trait, with no priming."""
    convs = []
    questions_flat = []
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


def main() -> None:
    args = parse_args()
    trait = args.trait
    hidden_layer = args.hidden_layer
    coef = args.coef
    hook_layer_idx = hidden_layer - 1   # forward-hook block index

    vec_path = preflight(trait, hidden_layer)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    artifact = load_trait(trait, version="eval")
    print(f"trait={trait}  eval questions={len(artifact.questions)}")

    full_stack = torch.load(vec_path, map_location="cpu")
    print(f"loaded vector stack {tuple(full_stack.shape)} from {vec_path}")
    vector = full_stack[hidden_layer]
    print(f"using output_hidden_states[{hidden_layer}] -> hook on block {hook_layer_idx}, coeff={coef}")

    model, tok = load_hf_model(MODEL_NAME)

    convs, questions_flat = _build_eval_conversations(artifact, N_PER_QUESTION)
    print(f"total generations per condition: {len(convs)}")

    print("\n=== baseline (no steering) ===")
    _, base_answers = generate_batch(
        model, tok, convs,
        max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, batch_size=BATCH_SIZE,
        steering=None,
    )
    base_trait, base_coh = _judge_run(JUDGE_MODEL, artifact.eval_prompt, questions_flat, base_answers, MAX_CONCURRENT_JUDGES)

    print(f"\n=== steered (coef={coef}, layer_idx={hook_layer_idx}, response-only) ===")
    _, steer_answers = generate_batch(
        model, tok, convs,
        max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, batch_size=BATCH_SIZE,
        steering=(vector, hook_layer_idx, coef, "response"),
    )
    steer_trait, steer_coh = _judge_run(JUDGE_MODEL, artifact.eval_prompt, questions_flat, steer_answers, MAX_CONCURRENT_JUDGES)

    df = pd.DataFrame({
        "question": questions_flat,
        "baseline_answer": base_answers,
        "baseline_trait": base_trait,
        "baseline_coherence": base_coh,
        "steered_answer": steer_answers,
        "steered_trait": steer_trait,
        "steered_coherence": steer_coh,
    })
    out_csv = OUT_DIR / f"{trait}_steer_response_layer{hidden_layer}_coef{coef}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")

    def _mean(col):
        valid = [x for x in col if x is not None and pd.notna(x)]
        return sum(valid) / len(valid) if valid else float("nan")

    print("\n=== summary (means over valid scores) ===")
    print(f"  baseline:  trait={_mean(base_trait):6.2f}   coherence={_mean(base_coh):6.2f}")
    print(f"  steered:   trait={_mean(steer_trait):6.2f}   coherence={_mean(steer_coh):6.2f}")
    print(f"  delta trait: {_mean(steer_trait) - _mean(base_trait):+.2f}")


if __name__ == "__main__":
    main()
