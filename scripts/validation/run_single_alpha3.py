"""
Extends the E10.3 single-vector α-sweep with α=3 (Riccardo, Pilot 2).

Phase 15 follow-up: E10.3's α grid is {2, 4, 6, 8}. The dramatic gap between α=2
(mean Δ_trait=+10, coh=94) and α=4 (mean Δ_trait=+46, coh=78) is suggestive of a
phase-shift region. α=3 probes the middle.

Writes one CSV per trait matching the existing alpha_sweep_l17 schema, but with
trait/coherence left as NaN (cluster has no outbound network for OpenAI calls;
judge stage runs on laptop afterward via run_single_alpha3_judge_local.py).

Output:
    results/alpha_sweep_l17/Llama-3.1-8B-Instruct/{trait}_unit_alpha3.0.csv

Idempotent — skips traits whose CSV already exists.

Run:
    COMPOSITION_PILOT_MODE=generate python -m scripts.validation.run_single_alpha3
"""

from __future__ import annotations

import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from src.extraction.generation import generate_batch
from src.extraction.trait_data import load_trait
from src.inference.hf_model import load_hf_model

load_dotenv()

# Config — held identical to E10.3 except for the single new α value.
TRAITS = [
    "apathetic", "evil", "hallucinating", "humorous", "impolite",
    "sycophantic", "power_seeking", "confidence", "formality",
]
POLARITY_INVERTED: set[str] = {"power_seeking"}

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HIDDEN_LAYER = 17
HOOK_LAYER_IDX = HIDDEN_LAYER - 1
ALPHA = 3.0

N_PER_QUESTION = 5
MAX_NEW_TOKENS = 600
TEMPERATURE = 1.0
BATCH_SIZE = 8

VECTOR_OUTPUT_DIR = Path("results/persona_vectors/Llama-3.1-8B-Instruct")
EVAL_OUTPUT_DIR = Path("results/alpha_sweep_l17/Llama-3.1-8B-Instruct")
LOGS_DIR = Path("logs")


def _unit_steer_csv_path(trait: str, alpha: float) -> Path:
    return EVAL_OUTPUT_DIR / f"{trait}_unit_alpha{alpha}.csv"


def _load_unit_vector(trait: str) -> torch.Tensor:
    vec_path = VECTOR_OUTPUT_DIR / f"{trait}_response_avg_diff.pt"
    if not vec_path.exists():
        raise FileNotFoundError(f"Vector not found at {vec_path}")
    stack = torch.load(vec_path, map_location="cpu")
    v = stack[HIDDEN_LAYER]
    if trait in POLARITY_INVERTED:
        v = -v
    return v / v.norm()


def _build_eval_conversations(questions, n_per_question):
    convs, qflat = [], []
    for q in questions:
        for _ in range(n_per_question):
            convs.append([{"role": "user", "content": q}])
            qflat.append(q)
    return convs, qflat


def _generate_csv(out_csv: Path, model, tok, artifact, unit_vector, alpha, log_path):
    """Same schema as run_alpha_sweep_l17 (`question, answer, trait, coherence`)
    but with trait/coherence left NaN. Laptop judge stage fills them in.
    """
    if out_csv.exists():
        return
    convs, qflat = _build_eval_conversations(artifact.questions, N_PER_QUESTION)
    with log_path.open("w", buffering=1) as fh, redirect_stdout(fh), redirect_stderr(fh):
        print(
            f"trait={artifact.trait}  α_unit={alpha}  "
            f"layer={HIDDEN_LAYER} (hook block {HOOK_LAYER_IDX})  "
            f"questions={len(qflat)}"
        )
        _, answers = generate_batch(
            model, tok, convs,
            max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
            batch_size=BATCH_SIZE,
            steering=(unit_vector, HOOK_LAYER_IDX, alpha, "response"),
        )
    df = pd.DataFrame({
        "question": qflat,
        "answer": answers,
        "trait": pd.array([pd.NA] * len(answers), dtype="Float64"),
        "coherence": pd.array([pd.NA] * len(answers), dtype="Float64"),
    })
    df.to_csv(out_csv, index=False)


def main():
    mode = os.environ.get("COMPOSITION_PILOT_MODE", "generate").lower()
    if mode not in {"generate", "judge", "full"}:
        raise SystemExit(f"COMPOSITION_PILOT_MODE must be generate|judge|full")
    if mode == "judge":
        print("This script is generate-only. Run the laptop judge wrapper instead:")
        print("  python -m scripts.validation.run_single_alpha3_judge_local")
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} ...")
    model, tok = load_hf_model(MODEL_NAME)
    print(
        f"Model loaded: hidden={model.config.hidden_size}  "
        f"n_layers={model.config.num_hidden_layers}\n"
    )

    print(f"Generating α={ALPHA} single-vector responses for {len(TRAITS)} traits\n")

    for i, trait in enumerate(TRAITS, 1):
        out_csv = _unit_steer_csv_path(trait, ALPHA)
        print(f"[{i}/{len(TRAITS)}] {trait:<14} -> {out_csv.name}")
        if out_csv.exists():
            print(f"  exists, skipping")
            continue
        v = _load_unit_vector(trait)
        artifact = load_trait(trait, version="eval")
        log_path = LOGS_DIR / f"single_alpha3_{trait}.log"
        _generate_csv(out_csv, model, tok, artifact, v, ALPHA, log_path)
        print(f"  {len(pd.read_csv(out_csv))} rows written")

    print(f"\nDone. Laptop next: python -m scripts.validation.run_single_alpha3_judge_local")


if __name__ == "__main__":
    main()
