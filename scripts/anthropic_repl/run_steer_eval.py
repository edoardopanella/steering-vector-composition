"""
Stage 3 of the Anthropic persona-vectors replication: steered-eval sanity check.

Loads response_avg_diff.pt[layer], generates on the held-out trait_data_eval set
both with and without steering, and judges both with the trait + coherence judges.
Acceptance: trait score under steering should be substantially higher than baseline,
mirroring Figure 2 of the paper.

Run:
    python -m scripts.anthropic_repl.run_steer_eval
"""

from __future__ import annotations

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
from src.anthropic_repl.trait_data import load_trait
from src.judge import OpenAiJudge

load_dotenv()

# --- config ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TRAIT = "evil"
JUDGE_MODEL = "gpt-4.1-mini"

# Chen et al. 2025, §B.4 (paper p.28-29 + §B.4 prose): "layer 16 is the most informative
# for all three traits" on Llama-3.1-8B-Instruct. Paper also clarifies that layer
# indexing is 1-based and refers to the *output* of that layer. So "layer 16
# activation" = output_hidden_states[16] (the residual stream after block 15 in
# HF's 0-indexed model.model.layers). Our forward hook fires on the output of a
# block, so we hook block 15.
HIDDEN_LAYER = 16      # output_hidden_states index — paper's "layer 16"
HOOK_LAYER_IDX = HIDDEN_LAYER - 1   # forward-hook target = block 15 (0-indexed)
COEFF = 2.0            # paper / repo eval_steering.sh use ±1.5–2.0 on raw vectors

N_PER_QUESTION = 5
MAX_NEW_TOKENS = 600
TEMPERATURE = 1.0
BATCH_SIZE = 8
# Lowered from 50 after stage 1 hit OpenAI TPM (200K/min) + RPM (500/min) caps
# during judging, losing ~9% of scores. With ~700 tokens/call, 4-5 in flight
# stays below TPM and RPM. Worst case: judging takes a few extra minutes; better
# than dropping scores.
MAX_CONCURRENT_JUDGES = 5

VECTORS_DIR = Path("results/anthropic_repl/persona_vectors") / MODEL_NAME.split("/")[-1]
OUT_DIR = Path("results/anthropic_repl/eval_persona_eval") / MODEL_NAME.split("/")[-1]
# --------------


def _build_eval_conversations(artifact, n_per_question):
    """Use plain user prompts (no system instruction) — eval set tests whether the
    vector itself elicits the trait, with no priming."""
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = load_trait(TRAIT, version="eval")
    print(f"trait={TRAIT}  eval questions={len(artifact.questions)}")

    vec_path = VECTORS_DIR / f"{TRAIT}_response_avg_diff.pt"
    assert vec_path.exists(), f"missing {vec_path} — run run_build_vector.py first"
    full_stack = torch.load(vec_path, map_location="cpu")
    print(f"loaded vector stack {tuple(full_stack.shape)} from {vec_path}")
    vector = full_stack[HIDDEN_LAYER]
    print(f"using output_hidden_states[{HIDDEN_LAYER}] -> hook on block {HOOK_LAYER_IDX}, coeff={COEFF}")

    model, tok = load_hf_model(MODEL_NAME)

    convs, questions_flat = _build_eval_conversations(artifact, N_PER_QUESTION)
    print(f"total generations per condition: {len(convs)}")

    # Baseline (unsteered)
    print("\n=== baseline (no steering) ===")
    _, base_answers = generate_batch(
        model, tok, convs,
        max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, batch_size=BATCH_SIZE,
        steering=None,
    )
    base_trait, base_coh = _judge_run(JUDGE_MODEL, artifact.eval_prompt, questions_flat, base_answers, MAX_CONCURRENT_JUDGES)

    # Steered
    print(f"\n=== steered (coef={COEFF}, layer_idx={HOOK_LAYER_IDX}, response-only) ===")
    _, steer_answers = generate_batch(
        model, tok, convs,
        max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, batch_size=BATCH_SIZE,
        steering=(vector, HOOK_LAYER_IDX, COEFF, "response"),
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
    out_csv = OUT_DIR / f"{TRAIT}_steer_response_layer{HIDDEN_LAYER}_coef{COEFF}.csv"
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
