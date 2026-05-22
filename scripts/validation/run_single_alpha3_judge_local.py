"""
Laptop judge stage for the α=3 single-vector extension to E10.3 (Pilot 2).

For each of the 9 traits, reads {trait}_unit_alpha3.0.csv from
results/alpha_sweep_l17/Llama-3.1-8B-Instruct/, calls the trait + coherence
judges on rows where trait is NaN, writes back. Idempotent.

Run:
    python -m scripts.validation.run_single_alpha3_judge_local
"""

from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.extraction.generation import COHERENCE_PROMPT, _judge_all
from src.extraction.trait_data import load_trait
from src.judge import OpenAiJudge

from scripts.validation.run_single_alpha3 import (
    ALPHA,
    LOGS_DIR,
    TRAITS,
    _unit_steer_csv_path,
)

load_dotenv()

JUDGE_MODEL = "gpt-4.1-mini"
MAX_CONCURRENT_JUDGES = 5


def _judge_csv(out_csv: Path, eval_prompt: str, progress_tag: str, log_path: Path) -> None:
    if not out_csv.exists():
        return
    df = pd.read_csv(out_csv)
    mask = df["trait"].isna()
    if not mask.any():
        return
    questions = df.loc[mask, "question"].astype(str).tolist()
    answers = df.loc[mask, "answer"].astype(str).fillna("").tolist()

    trait_judge = OpenAiJudge(JUDGE_MODEL, eval_prompt, eval_type="0_100")
    coh_judge = OpenAiJudge(JUDGE_MODEL, COHERENCE_PROMPT, eval_type="0_100")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    with log_path.open("a", buffering=1) as fh, redirect_stdout(fh), redirect_stderr(fh):
        print(f"[judge] {progress_tag} rows_to_judge={len(questions)}")
        try:
            trait_scores = loop.run_until_complete(
                _judge_all(trait_judge, questions, answers, MAX_CONCURRENT_JUDGES,
                           progress_label=f"{progress_tag} trait", progress_every=25)
            )
            coh_scores = loop.run_until_complete(
                _judge_all(coh_judge, questions, answers, MAX_CONCURRENT_JUDGES,
                           progress_label=f"{progress_tag} coherence", progress_every=25)
            )
        finally:
            loop.close()

    def _to_nan(xs):
        return [float("nan") if x is None else float(x) for x in xs]

    df.loc[mask, "trait"] = _to_nan(trait_scores)
    df.loc[mask, "coherence"] = _to_nan(coh_scores)
    df.to_csv(out_csv, index=False)


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"LOCAL JUDGE STAGE — single-vector α={ALPHA}")
    print(f"  CSV dir : results/alpha_sweep_l17/Llama-3.1-8B-Instruct")
    print(f"  judge   : {JUDGE_MODEL}")
    print("=" * 72 + "\n")

    for i, trait in enumerate(TRAITS, 1):
        out_csv = _unit_steer_csv_path(trait, ALPHA)
        print(f"[{i}/{len(TRAITS)}] {trait:<14} -> {out_csv.name}")
        if not out_csv.exists():
            print(f"  SKIP: CSV missing")
            continue
        df = pd.read_csv(out_csv)
        scored = int(df["trait"].notna().sum()) if "trait" in df.columns else 0
        print(f"  {len(df)} rows, {scored} already scored")
        if scored == len(df):
            continue
        artifact = load_trait(trait, version="eval")
        _judge_csv(
            out_csv, artifact.eval_prompt,
            progress_tag=f"{trait} α={ALPHA}",
            log_path=LOGS_DIR / f"single_alpha3_{trait}.log",
        )
        df2 = pd.read_csv(out_csv)
        print(f"  -> {int(df2['trait'].notna().sum())} scored after pass")

    print("\nDone.")


if __name__ == "__main__":
    main()
