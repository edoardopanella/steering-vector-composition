"""
Paper-faithful replication of Chen et al. 2025 §B.4 layer selection on Llama-3.1-8B-Instruct.

The team's E9 sweep (run_layer_selection_all.py) found shared L*=17. This script
re-runs layer selection with the choices realigned to the paper's stated protocol,
to test whether our infrastructure reproduces the paper's L*=16 finding before we
move on to defending L=17 as a deliberate downstream choice.

Differences from run_layer_selection_all.py — every one of these flips an E9 choice
back to what the paper actually does:

    Trait set
        E9: 9 traits (6 paper + 3 project-generated).
        Here: only the 3 paper traits {evil, sycophantic, hallucinating} that
        the §B.4 "L=16 for all three" claim is actually about.

    Selection metric
        E9: argmax of (steer_trait - baseline_trait).
        Here: argmax of absolute steer_trait. Paper §B.4: "select the layer that
        elicits the highest trait expression score." Not a delta, not relative
        to baseline.

    Coherence filter
        E9: coh_floor=50 with fallback.
        Here: none. Paper §B.4 imposes no coherence constraint.

    Per-trait vs shared
        E9: averages Δ_trait across 9 traits to pick a shared L*.
        Here: per-trait L*, no averaging. Paper picks per trait, then reports
        the three picks happened to coincide at 16.

    Coefficient
        E9: α=2.0 fixed across all traits and layers.
        Here: trait-specific α matching the values the paper reports in Tables
        8/9/10 of the appendix — α=2.5 for evil & sycophantic, α=1.5 for
        hallucinating. Paper Figure 13 shows multiple curves per coefficient;
        we run one sweep per trait at the trait's paper-canonical α.

    Rollouts per question
        E9: N_PER_QUESTION=1 with TEMPERATURE=1.0 (single stochastic sample,
        ~σ_judge per (trait, layer) point — too noisy to distinguish adjacent
        layers like 16 vs 17).
        Here: N_PER_QUESTION=5. Paper §3.1 generates 10 rollouts elsewhere;
        N=5 is a reasonable cost-bounded compromise that still reduces SE per
        (trait, layer) to ~σ/sqrt(100) ≈ 1–3 judge points.

Re-uses (unchanged): the same vector files, generation stack, hook, judge calls,
trait+coherence rubrics, and chat templating that E9 used. Anything that affects
"are we steering at the right place" stays identical to the rest of the project's
anthropic_repl pipeline; only the *selection rule* and *sample size* change.

Inputs:
    results/persona_vectors/Llama-3.1-8B-Instruct/{trait}_response_avg_diff.pt
    -> stack [33, 4096], indexed by hidden_layer ∈ [1, 32].

Outputs:
    results/eval_persona_eval_paper_repl/Llama-3.1-8B-Instruct/
        {trait}_baseline.csv                                      (1 per trait)
        {trait}_layer{L}_coef{coef}_steer_response.csv            (1 per (trait, L))
    results/layer_selection_paper_repl.json                       (per-trait picks)

Run:
    python -m scripts.layer_selection.run_paper_repl
"""

from __future__ import annotations

import asyncio
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from src.extraction.generation import COHERENCE_PROMPT, _judge_all, generate_batch
from src.extraction.trait_data import load_trait
from src.inference.hf_model import load_hf_model
from src.judge import OpenAiJudge

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Paper's three Llama-3.1-8B-Instruct traits with the coefficients reported in
# Tables 8 (evil), 9 (sycophantic), 10 (hallucinating) of the appendix. Paper
# §B.4 uses a single coefficient per trait per sweep; the per-trait α reflects
# what the paper actually used at the layer they selected.
TRAITS_AND_COEFFS = [
    ("evil",          2.5),
    ("sycophantic",   2.5),
    ("hallucinating", 1.5),
]

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "gpt-4.1-mini"

# hidden_layer = output_hidden_states index. 0 = embeddings (no preceding block
# to hook), so we sweep 1..32 → hook on block 0..31. 1-indexed convention
# matches paper §B.4 footnote ("layer 16 = the 16th transformer block's output").
HIDDEN_LAYERS = list(range(1, 33))

# Sample size. E9 used 1, which produced single-shot stochastic estimates. The
# paper's other experiments (§3.1) use 10 rollouts. N=5 brings per-(trait, layer)
# SE down to ~σ_judge/sqrt(20*5) ≈ 1–3 trait-score units, enough to actually
# distinguish layer 16 from layer 17 if there is a real difference.
N_PER_QUESTION = 5
MAX_NEW_TOKENS = 600
TEMPERATURE = 1.0
BATCH_SIZE = 8
MAX_CONCURRENT_JUDGES = 5

VECTOR_DIR = Path("results/persona_vectors") / MODEL_NAME.split("/")[-1]
SWEEP_OUT_DIR = (
    Path("results/eval_persona_eval_paper_repl") / MODEL_NAME.split("/")[-1]
)
SUMMARY_PATH = Path("results/layer_selection_paper_repl.json")
LOGS_DIR = Path("logs")
# ---------------------------------------------------------------------------


def build_eval_conversations(artifact, n_per_question: int):
    convs, questions_flat = [], []
    for q in artifact.questions:
        for _ in range(n_per_question):
            convs.append([{"role": "user", "content": q}])
            questions_flat.append(q)
    return convs, questions_flat


def judge_run(judge_model, eval_prompt, questions, answers, max_concurrent):
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


def mean_valid(col) -> float:
    valid = [x for x in col if x is not None and pd.notna(x)]
    return sum(valid) / len(valid) if valid else float("nan")


def run_baseline(trait, model, tok, artifact, log_path: Path) -> Path:
    SWEEP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = SWEEP_OUT_DIR / f"{trait}_baseline.csv"
    if out_csv.exists():
        return out_csv

    convs, questions_flat = build_eval_conversations(artifact, N_PER_QUESTION)
    with log_path.open("w") as fh:
        with redirect_stdout(fh), redirect_stderr(fh):
            print(f"trait={trait}  baseline  questions={len(questions_flat)}")
            _, answers = generate_batch(
                model, tok, convs,
                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, batch_size=BATCH_SIZE,
                steering=None,
            )
            trait_scores, coh_scores = judge_run(
                JUDGE_MODEL, artifact.eval_prompt, questions_flat, answers, MAX_CONCURRENT_JUDGES
            )
            df = pd.DataFrame({
                "question": questions_flat,
                "answer": answers,
                "trait": trait_scores,
                "coherence": coh_scores,
            })
            df.to_csv(out_csv, index=False)
    return out_csv


def run_steered_layer(
    trait, hidden_layer, coef, model, tok, artifact, vector_stack, log_path: Path,
) -> Path:
    SWEEP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = SWEEP_OUT_DIR / f"{trait}_layer{hidden_layer}_coef{coef}_steer_response.csv"
    if out_csv.exists():
        return out_csv

    vector = vector_stack[hidden_layer]
    hook_layer_idx = hidden_layer - 1
    convs, questions_flat = build_eval_conversations(artifact, N_PER_QUESTION)
    with log_path.open("w") as fh:
        with redirect_stdout(fh), redirect_stderr(fh):
            print(
                f"trait={trait}  hidden_layer={hidden_layer}  "
                f"hook_block={hook_layer_idx}  coef={coef}"
            )
            _, answers = generate_batch(
                model, tok, convs,
                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, batch_size=BATCH_SIZE,
                steering=(vector, hook_layer_idx, coef, "response"),
            )
            trait_scores, coh_scores = judge_run(
                JUDGE_MODEL, artifact.eval_prompt, questions_flat, answers, MAX_CONCURRENT_JUDGES
            )
            df = pd.DataFrame({
                "question": questions_flat,
                "answer": answers,
                "trait": trait_scores,
                "coherence": coh_scores,
            })
            df.to_csv(out_csv, index=False)
    return out_csv


def summarise_csv(csv_path: Path) -> tuple[float, float]:
    df = pd.read_csv(csv_path)
    return mean_valid(df["trait"]), mean_valid(df["coherence"])


def pick_paper_l_star(layers_summary: dict[int, dict]) -> tuple[int, float]:
    """Paper §B.4 rule: argmax of absolute steer_trait. No coherence filter."""
    L_star = max(layers_summary, key=lambda L: layers_summary[L]["steer_trait"])
    return L_star, layers_summary[L_star]["steer_trait"]


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} …")
    model, tok = load_hf_model(MODEL_NAME)
    print(
        f"Model loaded: hidden={model.config.hidden_size}  "
        f"n_layers={model.config.num_hidden_layers}\n"
    )

    per_trait: dict[str, dict] = {}

    for ti, (trait, coef) in enumerate(TRAITS_AND_COEFFS, 1):
        print(f"\n[{ti}/{len(TRAITS_AND_COEFFS)}] trait={trait}  coef={coef}")

        vec_path = VECTOR_DIR / f"{trait}_response_avg_diff.pt"
        if not vec_path.exists():
            print(f"  WARNING: vector not found at {vec_path} — skipping")
            continue

        vector_stack = torch.load(vec_path, map_location="cpu", weights_only=True)
        if vector_stack.shape != torch.Size([33, 4096]):
            print(f"  WARNING: unexpected vector shape {tuple(vector_stack.shape)} — skipping")
            continue

        artifact = load_trait(trait, version="eval")

        base_log = LOGS_DIR / f"paperrepl_{trait}_baseline.log"
        print(f"  baseline … ", end="", flush=True)
        base_csv = run_baseline(trait, model, tok, artifact, base_log)
        base_trait_mean, base_coh_mean = summarise_csv(base_csv)
        print(f"trait={base_trait_mean:.2f}  coh={base_coh_mean:.2f}  (csv: {base_csv.name})")

        layers_summary: dict[int, dict] = {}
        for li, hidden_layer in enumerate(HIDDEN_LAYERS, 1):
            steer_log = LOGS_DIR / f"paperrepl_{trait}_L{hidden_layer}_a{coef}.log"
            print(f"  layer {hidden_layer:2d} [{li}/{len(HIDDEN_LAYERS)}] … ", end="", flush=True)
            steer_csv = run_steered_layer(
                trait, hidden_layer, coef, model, tok, artifact, vector_stack, steer_log
            )
            steer_trait_mean, steer_coh_mean = summarise_csv(steer_csv)
            layers_summary[hidden_layer] = {
                "steer_trait": steer_trait_mean,
                "steer_coh": steer_coh_mean,
                "delta_trait": steer_trait_mean - base_trait_mean,
                "delta_coh": steer_coh_mean - base_coh_mean,
            }
            print(
                f"trait={steer_trait_mean:6.2f}  coh={steer_coh_mean:6.2f}  "
                f"Δtrait={layers_summary[hidden_layer]['delta_trait']:+7.2f}"
            )

        L_star, L_star_trait = pick_paper_l_star(layers_summary)
        per_trait[trait] = {
            "coef": coef,
            "baseline_trait": base_trait_mean,
            "baseline_coh": base_coh_mean,
            "layers": layers_summary,
            "L_star": L_star,
            "L_star_steer_trait": L_star_trait,
        }
        paper_pick = 16
        L_star_vs_paper = (
            "MATCH" if L_star == paper_pick else f"DIFFERS (paper L=16, ours L={L_star})"
        )
        print(
            f"  -> per-trait L* = {L_star}  (steer_trait = {L_star_trait:.2f})  "
            f"[paper says L=16: {L_star_vs_paper}]"
        )

        with SUMMARY_PATH.open("w") as fh:
            json.dump(_serialise(per_trait), fh, indent=2)

    final = {
        "config": {
            "model": MODEL_NAME,
            "judge_model": JUDGE_MODEL,
            "traits_and_coefs": TRAITS_AND_COEFFS,
            "hidden_layers": HIDDEN_LAYERS,
            "n_per_question": N_PER_QUESTION,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "selection_rule": "argmax(steer_trait), no coherence filter, per-trait",
            "paper_reference": "Chen et al. 2025, arXiv:2507.21509, Appendix §B.4",
            "paper_pick_for_all_three": 16,
        },
        "per_trait": _serialise(per_trait),
    }
    with SUMMARY_PATH.open("w") as fh:
        json.dump(final, fh, indent=2)

    # ---------------------------------------------------------------------------
    # End-of-run table
    # ---------------------------------------------------------------------------
    print()
    print(
        f"{'trait':<14} {'coef':>4} {'L*':>4} {'paper':>6} {'match?':>7} "
        f"{'base_tr':>8} {'L*_tr':>7} {'L*_coh':>7}"
    )
    print("-" * 70)
    matches = 0
    for t, s in per_trait.items():
        L = s["L_star"]
        st = s["layers"][L]
        match = "✓" if L == 16 else "✗"
        if L == 16:
            matches += 1
        print(
            f"{t:<14} {s['coef']:>4.1f} {L:>4d} {16:>6d} {match:>7} "
            f"{s['baseline_trait']:>8.2f} {st['steer_trait']:>7.2f} {st['steer_coh']:>7.2f}"
        )
    print()
    print(f"reproduction: {matches}/3 traits land at the paper's L=16")
    print(f"summary saved to {SUMMARY_PATH}")


def _serialise(per_trait: dict) -> dict:
    """Convert int layer keys to str for JSON."""
    out = {}
    for t, s in per_trait.items():
        out[t] = {
            "coef": s["coef"],
            "baseline_trait": s["baseline_trait"],
            "baseline_coh": s["baseline_coh"],
            "layers": {str(L): v for L, v in s["layers"].items()},
            "L_star": s["L_star"],
            "L_star_steer_trait": s["L_star_steer_trait"],
        }
    return out


if __name__ == "__main__":
    main()
