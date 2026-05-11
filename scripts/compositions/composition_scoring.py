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

from src.composition.joint_injection import (
    compose_steering_vector,
    project_activation,
    trajectory_response_avg,
)
from src.extraction.generation import COHERENCE_PROMPT, _judge_all, generate_batch
from src.geometry.clusters import pair_cluster_status
from src.geometry.pair_strat import assign_stratum
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

# --- Phase 2 trajectory dataset ---------------------------------------------
# Roadmap §5: cache projection trajectories for every (pair, setting, prompt,
# layer, behaviour) at the operating point. Settings reduced to (1,0)/(0,1)/(1,1)
# — the antipodal probes (1,-1)/(-1,1) are RQ1 robustness, not mechanism.
TRAJECTORY_SETTINGS: list[tuple[int, int]] = [(1, 0), (0, 1), (1, 1)]
TRAJECTORY_OUT_DIR = Path("results/composition_trajectories_l17/Llama-3.1-8B-Instruct")
TRAJECTORY_AGG_PARQUET = Path("results/composition_trajectories_l17.parquet")

# Regime classification thresholds — applied to ratio(Δ_joint / Δ_single) per axis.
# Δ_joint = composition setting (1,1) Δ on a given trait vs baseline (0,0);
# Δ_single = single-vector setting (1,0) or (0,1) Δ on the matching trait.
REGIME_ADDITIVE_LO = 0.7   # both ratios in [LO, HI] -> additive
REGIME_ADDITIVE_HI = 1.3
REGIME_DOMINANT_LO = 0.7   # one ratio >= LO and other <= HI -> dominant
REGIME_DOMINANT_HI = 0.3
REGIME_SUPPRESSIVE_MAX = 0.5  # both ratios < this -> suppressive
REGIME_EMERGENT_MIN = 1.3     # both ratios > this -> emergent
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
    """Steering vector with alpha-sweep polarity convention applied. Used to
    build the δ injected at L*-1 — matches the scoring CSV polarity so reused
    (1,1) completions stay consistent with the recomputed teacher-force δ.
    """
    vec_path = VECTOR_OUTPUT_DIR / f"{trait}_response_avg_diff.pt"
    if not vec_path.exists():
        raise FileNotFoundError(f"Vector not found at {vec_path}")
    stack = torch.load(vec_path, map_location="cpu")
    v = stack[HIDDEN_LAYER]
    if trait in POLARITY_INVERTED:
        v = -v
    return v / v.norm()


def _load_unit_vector_raw(trait: str) -> torch.Tensor:
    """Raw-direction unit vector (NO polarity flip). Used as the projection
    axis in Phase 2 trajectory rows so π values + cosines match the E11 pilot
    + paper Figure 20 / E7.4 conventions, which never flip `power_seeking`.
    """
    vec_path = VECTOR_OUTPUT_DIR / f"{trait}_response_avg_diff.pt"
    if not vec_path.exists():
        raise FileNotFoundError(f"Vector not found at {vec_path}")
    stack = torch.load(vec_path, map_location="cpu")
    v = stack[HIDDEN_LAYER]
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
    progress_tag: str = "",
) -> tuple[list, list, list]:
    """Three judges (trait_a, trait_b, coherence), each scored 0-100. The
    `progress_tag` (e.g. ``"apathetic+confidence baseline"``) prefixes the
    `[judge] ... done/total` lines that _judge_all emits to the main SLURM
    stdout (visible via `tail -f /home/.../composition_scoring_<jobid>.out`)
    so progress is observable without opening per-pair logs.
    """
    judge_a = OpenAiJudge(judge_model, eval_prompt_a, eval_type="0_100")
    judge_b = OpenAiJudge(judge_model, eval_prompt_b, eval_type="0_100")
    judge_coh = OpenAiJudge(judge_model, COHERENCE_PROMPT, eval_type="0_100")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores_a = loop.run_until_complete(_judge_all(
            judge_a, questions, answers, max_concurrent,
            progress_label=f"{progress_tag} trait_a", progress_every=25,
        ))
        scores_b = loop.run_until_complete(_judge_all(
            judge_b, questions, answers, max_concurrent,
            progress_label=f"{progress_tag} trait_b", progress_every=25,
        ))
        scores_coh = loop.run_until_complete(_judge_all(
            judge_coh, questions, answers, max_concurrent,
            progress_label=f"{progress_tag} coherence", progress_every=25,
        ))
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
    with log_path.open("w", buffering=1) as fh:
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
                progress_tag=f"{trait_a}+{trait_b} baseline",
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
    with log_path.open("w", buffering=1) as fh:
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
                progress_tag=f"{trait_a}+{trait_b} joint α={alpha}",
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


# === Phase 2: trajectory dataset ===========================================

def _single_csv_path(trait_a: str, trait_b: str, which: str, alpha: float) -> Path:
    a, b = sorted([trait_a, trait_b])
    return SCORES_OUTPUT_DIR / f"{a}__{b}_single_{which}_alpha{alpha}.csv"


def _trajectory_pair_parquet(trait_a: str, trait_b: str) -> Path:
    a, b = sorted([trait_a, trait_b])
    return TRAJECTORY_OUT_DIR / f"{a}__{b}.parquet"


def _run_single_steered_composition(
    trait_a: str, trait_b: str, w_a: int, w_b: int, artifact: dict, alpha: float,
    model, tok, v_a_unit: torch.Tensor, v_b_unit: torch.Tensor, log_path: Path,
) -> dict:
    """Generate + judge a single-vector setting (1,0) or (0,1) on the pair's
    composition_eval questions. Idempotent. Same δ math as the joint runner —
    one weight set to 0 — so the steering hook applies α·v on the active trait
    only. Judge stage scores trait_a, trait_b, coherence on every completion so
    Δ_a from setting (1,0) and Δ_b from setting (0,1) feed the regime
    classifier.
    """
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    which = "a" if (w_a, w_b) == (1, 0) else "b"
    out_csv = _single_csv_path(trait_a, trait_b, which, alpha)
    if out_csv.exists():
        return _summarise_df(pd.read_csv(out_csv))

    delta = compose_steering_vector(
        [(v_a_unit, float(w_a)), (v_b_unit, float(w_b))],
        alpha=alpha, normalize=False,
    )
    convs, questions_flat = _build_eval_conversations(artifact["questions"], N_PER_QUESTION)
    with log_path.open("w", buffering=1) as fh:
        with redirect_stdout(fh), redirect_stderr(fh):
            print(
                f"pair={trait_a}+{trait_b}  single ({w_a},{w_b}) α={alpha}  "
                f"layer={HIDDEN_LAYER}  questions={len(questions_flat)}"
            )
            _, answers = generate_batch(
                model, tok, convs,
                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, batch_size=BATCH_SIZE,
                steering=(delta, HOOK_LAYER_IDX, 1.0, "response"),
            )
            scores_a, scores_b, scores_coh = _judge_run_composition(
                JUDGE_MODEL,
                artifact["eval_prompt_a"], artifact["eval_prompt_b"],
                questions_flat, answers, MAX_CONCURRENT_JUDGES,
                progress_tag=f"{trait_a}+{trait_b} single_{which} α={alpha}",
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


def _classify_regime(
    delta_a_joint: float, delta_b_joint: float,
    delta_a_single: float, delta_b_single: float,
) -> str:
    """Return regime label from joint vs single per-trait Δ ratios.

    delta_a_joint / delta_a_single = ratio_a (1.0 = additive on a-axis).
    """
    eps = 1e-3
    ratio_a = delta_a_joint / delta_a_single if abs(delta_a_single) > eps else 0.0
    ratio_b = delta_b_joint / delta_b_single if abs(delta_b_single) > eps else 0.0

    if (REGIME_ADDITIVE_LO <= ratio_a <= REGIME_ADDITIVE_HI
            and REGIME_ADDITIVE_LO <= ratio_b <= REGIME_ADDITIVE_HI):
        return "additive"
    if ratio_a < REGIME_SUPPRESSIVE_MAX and ratio_b < REGIME_SUPPRESSIVE_MAX:
        return "suppressive"
    if ratio_a > REGIME_EMERGENT_MIN and ratio_b > REGIME_EMERGENT_MIN:
        return "emergent"
    if ((ratio_a >= REGIME_DOMINANT_LO and ratio_b <= REGIME_DOMINANT_HI)
            or (ratio_b >= REGIME_DOMINANT_LO and ratio_a <= REGIME_DOMINANT_HI)):
        return "dominant"
    return "mixed"


def _capture_trajectories_for_pair(
    pair_id: int, trait_a: str, trait_b: str,
    model, tok,
    v_a_steer: torch.Tensor, v_b_steer: torch.Tensor,
    v_a_proj: torch.Tensor, v_b_proj: torch.Tensor,
    layers: list[int], alpha: float,
) -> pd.DataFrame:
    """Phase 2 capture for one pair: read cached completions for each setting
    in TRAJECTORY_SETTINGS, teacher-force a trace through layers L ∈ `layers`
    with the matching δ injected at block (L*-1), project response-averaged
    activations onto the RAW-direction projection vectors (v_*_proj), emit one
    row per (setting, prompt_id, layer, behaviour). Per-pair Parquet is idempotent.

    Vector split (E11 polarity convention):
      v_*_steer — what scoring CSV used to build δ (alpha-sweep polarity, may
                  be flipped). Reused here so teacher-force δ matches generation
                  δ from the joint scoring CSV.
      v_*_proj  — raw direction (no flip). Projection axis matches the pilot's
                  reporting convention so cross-pair π values + cosines line up
                  with E11.4 / Figure 20.

    τ for L_div is NOT computed here — derived downstream via the R2 split-half
    bootstrap (see `_calibrate_tau_r2`) pooled over all 36 pairs.
    """
    out_path = _trajectory_pair_parquet(trait_a, trait_b)
    if out_path.exists():
        return pd.read_parquet(out_path)

    csv_per_setting = {
        (1, 0): _single_csv_path(trait_a, trait_b, "a", alpha),
        (0, 1): _single_csv_path(trait_a, trait_b, "b", alpha),
        (1, 1): _steered_csv_path(trait_a, trait_b, alpha),
    }
    for setting, p in csv_per_setting.items():
        if not p.exists():
            raise FileNotFoundError(
                f"trajectory capture needs CSV for setting {setting}: {p}"
            )

    rows: list[dict] = []
    for setting, csv_path in csv_per_setting.items():
        df = pd.read_csv(csv_path)
        w_a, w_b = setting
        delta = compose_steering_vector(
            [(v_a_steer, float(w_a)), (v_b_steer, float(w_b))],
            alpha=alpha, normalize=False,
        )
        for prompt_id, (q, ans) in enumerate(zip(df["question"], df["answer"])):
            question_id = prompt_id // N_PER_QUESTION
            completion_id = prompt_id % N_PER_QUESTION
            avg = trajectory_response_avg(
                model, tok, str(q), str(ans),
                layers_above=layers,
                delta_at_lstar=delta,
                layer_star=HIDDEN_LAYER,
            )
            for L in layers:
                h = avg[L]
                pi_a = float(project_activation(h, v_a_proj).item())
                pi_b = float(project_activation(h, v_b_proj).item())
                rows.append({
                    "pair_id": pair_id, "behaviour_index": 0,
                    "alpha_i": float(w_a) * alpha, "alpha_j": float(w_b) * alpha,
                    "prompt_id": prompt_id,
                    "question_id": question_id, "completion_id": completion_id,
                    "layer": int(L),
                    "projection_value": pi_a,
                })
                rows.append({
                    "pair_id": pair_id, "behaviour_index": 1,
                    "alpha_i": float(w_a) * alpha, "alpha_j": float(w_b) * alpha,
                    "prompt_id": prompt_id,
                    "question_id": question_id, "completion_id": completion_id,
                    "layer": int(L),
                    "projection_value": pi_b,
                })
    df_out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    return df_out


# === τ R2 calibration (Phase 2 pre-registered) =============================

# E11.5 verdict: original calibrate_tau (max layer-to-layer step in raw
# projection) measures noise of the wrong quantity (its scale tracks ‖h(L)‖).
# Replacement is R2: under a matched individual-steering condition, draw B
# split-halves of the per-prompt projections, take max_L |mean_A(L) − mean_B(L)|,
# pool across all (pair, setting, behaviour_axis), take 95th percentile,
# multiply by the roadmap §4 cushion factor 1.5. Pooling across all 36 pairs
# gives τ as a single cross-pair number (Phase 3 boxplot comparability).

TAU_RECIPE = "R2_split_half_bootstrap_q95_x1.5"
TAU_FACTOR = 1.5
TAU_BOOTSTRAP_DRAWS = 1000
TAU_Q = 0.95
TAU_OUT_PATH = Path("results/composition_trajectories_l17_tau.json")


def _calibrate_tau_r2(
    agg_df: pd.DataFrame,
    layers: list[int],
    alpha: float,
    seed: int = 42,
) -> dict:
    """Pool per-pair individual-steering trajectories. For each
    (pair_id, setting, behaviour_axis) draw `TAU_BOOTSTRAP_DRAWS` split-halves
    of the per-prompt projection samples, compute `max_L |mean_A(L) - mean_B(L)|`,
    pool, take `TAU_Q`-quantile, multiply by `TAU_FACTOR`.

    Individual-steering subset:
      setting (1,0), behaviour_index 0 (project on v_a_proj)
      setting (0,1), behaviour_index 1 (project on v_b_proj)
    Joint setting (1,1) excluded — that is the alternative the threshold gates.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    indiv_mask = (
        ((agg_df["alpha_i"] == alpha) & (agg_df["alpha_j"] == 0.0)
         & (agg_df["behaviour_index"] == 0))
        | ((agg_df["alpha_i"] == 0.0) & (agg_df["alpha_j"] == alpha)
           & (agg_df["behaviour_index"] == 1))
    )
    sub = agg_df.loc[indiv_mask]

    pooled_max_diffs: list[float] = []
    group_cols = ["pair_id", "alpha_i", "alpha_j", "behaviour_index"]
    for _, g in sub.groupby(group_cols, sort=False):
        # pivot: rows = prompt_id, cols = layer, values = projection_value
        wide = g.pivot_table(
            index="prompt_id", columns="layer", values="projection_value", aggfunc="mean"
        )
        wide = wide.reindex(columns=layers)
        arr = wide.to_numpy(dtype=float)            # [N_prompts, n_layers]
        n = arr.shape[0]
        if n < 2:
            continue
        half = n // 2
        for _ in range(TAU_BOOTSTRAP_DRAWS):
            idx = rng.permutation(n)
            a_mean = arr[idx[:half]].mean(axis=0)
            b_mean = arr[idx[half:half * 2]].mean(axis=0)
            pooled_max_diffs.append(float(np.max(np.abs(a_mean - b_mean))))

    if not pooled_max_diffs:
        return {
            "recipe": TAU_RECIPE, "tau": float("nan"),
            "n_groups": 0, "n_draws_total": 0,
            "factor": TAU_FACTOR, "q": TAU_Q, "bootstrap_draws_per_group": TAU_BOOTSTRAP_DRAWS,
            "note": "no individual-steering rows found",
        }
    arr = np.array(pooled_max_diffs)
    q_val = float(np.quantile(arr, TAU_Q))
    tau = TAU_FACTOR * q_val
    return {
        "recipe": TAU_RECIPE,
        "tau": tau,
        "factor": TAU_FACTOR,
        "q": TAU_Q,
        "q_value_pre_factor": q_val,
        "bootstrap_draws_per_group": TAU_BOOTSTRAP_DRAWS,
        "n_groups": int(sub.groupby(group_cols).ngroups),
        "n_draws_total": int(arr.size),
        "pooled_diff_min": float(arr.min()),
        "pooled_diff_max": float(arr.max()),
        "pooled_diff_mean": float(arr.mean()),
    }


def _l17_sanity_check(
    df_pair: pd.DataFrame, trait_a: str, trait_b: str,
    alpha: float, cos_ij: float,
) -> dict:
    """Closed form at L=L* on completion-divergence-noisy means:
    π_a^(1,1)(L*) − π_a^(1,0)(L*) ≈ α·cos(v_a, v_b)
    π_b^(1,1)(L*) − π_b^(0,1)(L*) ≈ α·cos(v_a, v_b)
    Reports observed vs predicted; tolerance is loose because completions
    differ across settings (E11.6).
    """
    L = HIDDEN_LAYER
    sub = df_pair[df_pair["layer"] == L]

    def _mean_at(behaviour_index, ai, aj):
        m = sub[(sub["behaviour_index"] == behaviour_index)
                & (sub["alpha_i"] == ai) & (sub["alpha_j"] == aj)]
        return float(m["projection_value"].mean()) if len(m) else float("nan")

    pi_a_10 = _mean_at(0, alpha, 0.0)
    pi_a_11 = _mean_at(0, alpha, alpha)
    pi_b_01 = _mean_at(1, 0.0, alpha)
    pi_b_11 = _mean_at(1, alpha, alpha)

    pred = alpha * cos_ij
    return {
        "layer": L, "alpha": alpha, "cos": cos_ij, "pred_a_cos": pred,
        "obs_pi_a_diff": pi_a_11 - pi_a_10,
        "obs_pi_b_diff": pi_b_11 - pi_b_01,
    }


# === Orchestration =========================================================

def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} ...")
    model, tok = load_hf_model(MODEL_NAME)
    print(f"Model loaded: hidden={model.config.hidden_size}  n_layers={model.config.num_hidden_layers}\n")

    # Trajectory layers: L >= L* up to the final block output (matches E11
    # pilot's `range(LAYER_STAR, n_layers + 1)`).
    n_layers = model.config.num_hidden_layers
    trajectory_layers = list(range(HIDDEN_LAYER, n_layers + 1))
    print(f"Trajectory layers: L ∈ [{trajectory_layers[0]}, {trajectory_layers[-1]}] "
          f"({len(trajectory_layers)} layers)")

    unit_vectors: dict[str, torch.Tensor] = {}        # for δ build (alpha-sweep polarity)
    unit_vectors_raw: dict[str, torch.Tensor] = {}    # for projection axis (no flip, pilot convention)
    for t in TRAITS:
        try:
            unit_vectors[t] = _load_unit_vector(t)
            unit_vectors_raw[t] = _load_unit_vector_raw(t)
            inv = " (inverted)" if t in POLARITY_INVERTED else ""
            print(f"  loaded unit vector for {t}{inv}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    pairs = _composition_pairs()
    print(f"\nEvaluating {len(pairs)} composition pairs at α={COMPOSITION_ALPHA}\n")

    summary_pairs: list[dict] = []
    pairs_meta: list[dict] = []           # one row per pair_id for parquet join
    trajectory_frames: list[pd.DataFrame] = []

    for i, (a, b) in enumerate(pairs, 1):
        pair_id = i - 1   # 0-indexed in parquet
        print(f"\n[{i}/{len(pairs)}] {a} + {b}  (pair_id={pair_id})")

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
        v_a_raw, v_b_raw = unit_vectors_raw[a], unit_vectors_raw[b]
        # Cosine reported on RAW directions to match E11.4 / Figure 20 convention.
        cos_ab = float((v_a_raw @ v_b_raw).item())

        # --- Baseline (0,0) ---
        base_log = LOGS_DIR / f"composition_{a}__{b}_baseline.log"
        print("  baseline … ", end="", flush=True)
        base = _run_baseline_composition(a, b, artifact, model, tok, base_log)
        print(
            f"trait_a={base['trait_a_mean']:.2f}  trait_b={base['trait_b_mean']:.2f}  "
            f"comp={base['composition_mean']:.2f}  coh={base['coherence_mean']:.2f}"
        )

        # --- Single (1,0) ---
        single_a_log = LOGS_DIR / f"composition_{a}__{b}_single_a_alpha{COMPOSITION_ALPHA}.log"
        print(f"  single_a (1,0) α={COMPOSITION_ALPHA} … ", end="", flush=True)
        single_a = _run_single_steered_composition(
            a, b, 1, 0, artifact, COMPOSITION_ALPHA, model, tok, v_a, v_b, single_a_log,
        )
        print(
            f"trait_a={single_a['trait_a_mean']:.2f}  trait_b={single_a['trait_b_mean']:.2f}  "
            f"coh={single_a['coherence_mean']:.2f}"
        )

        # --- Single (0,1) ---
        single_b_log = LOGS_DIR / f"composition_{a}__{b}_single_b_alpha{COMPOSITION_ALPHA}.log"
        print(f"  single_b (0,1) α={COMPOSITION_ALPHA} … ", end="", flush=True)
        single_b = _run_single_steered_composition(
            a, b, 0, 1, artifact, COMPOSITION_ALPHA, model, tok, v_a, v_b, single_b_log,
        )
        print(
            f"trait_a={single_b['trait_a_mean']:.2f}  trait_b={single_b['trait_b_mean']:.2f}  "
            f"coh={single_b['coherence_mean']:.2f}"
        )

        # --- Joint (1,1) ---
        steer_log = LOGS_DIR / f"composition_{a}__{b}_alpha{COMPOSITION_ALPHA}.log"
        print(f"  joint   (1,1) α={COMPOSITION_ALPHA} … ", end="", flush=True)
        steered = _run_joint_steered_composition(
            a, b, artifact, COMPOSITION_ALPHA, model, tok, v_a, v_b, steer_log,
        )
        delta_a_joint = steered["trait_a_mean"] - base["trait_a_mean"]
        delta_b_joint = steered["trait_b_mean"] - base["trait_b_mean"]
        delta_a_single = single_a["trait_a_mean"] - base["trait_a_mean"]
        delta_b_single = single_b["trait_b_mean"] - base["trait_b_mean"]
        delta_comp = steered["composition_mean"] - base["composition_mean"]
        delta_coh = steered["coherence_mean"] - base["coherence_mean"]
        print(
            f"trait_a={steered['trait_a_mean']:.2f} (Δ{delta_a_joint:+.2f})  "
            f"trait_b={steered['trait_b_mean']:.2f} (Δ{delta_b_joint:+.2f})  "
            f"comp={steered['composition_mean']:.2f} (Δ{delta_comp:+.2f})  "
            f"coh={steered['coherence_mean']:.2f} (Δ{delta_coh:+.2f})"
        )

        # --- Regime classification ---
        regime = _classify_regime(delta_a_joint, delta_b_joint, delta_a_single, delta_b_single)
        print(
            f"  regime={regime}  "
            f"ratio_a={delta_a_joint / delta_a_single if abs(delta_a_single) > 1e-3 else float('nan'):+.2f}  "
            f"ratio_b={delta_b_joint / delta_b_single if abs(delta_b_single) > 1e-3 else float('nan'):+.2f}"
        )

        # --- Phase 2 trajectory capture ---
        print(f"  trajectory capture … ", end="", flush=True)
        df_traj = _capture_trajectories_for_pair(
            pair_id, a, b, model, tok,
            v_a, v_b,            # steering δ (alpha-sweep polarity)
            v_a_raw, v_b_raw,    # projection axis (raw direction)
            trajectory_layers, COMPOSITION_ALPHA,
        )
        print(f"{len(df_traj)} rows -> {_trajectory_pair_parquet(a, b)}")

        # --- L=L* sanity check ---
        sanity = _l17_sanity_check(df_traj, a, b, COMPOSITION_ALPHA, cos_ab)
        print(
            f"  L={sanity['layer']} sanity: cos={cos_ab:+.3f}  "
            f"pred α·cos={sanity['pred_a_cos']:+.3f}  "
            f"obs Δπ_a={sanity['obs_pi_a_diff']:+.3f}  "
            f"obs Δπ_b={sanity['obs_pi_b_diff']:+.3f}"
        )

        # --- Pair metadata for the aggregate parquet ---
        cluster_status = pair_cluster_status(a, b)
        pairs_meta.append({
            "pair_id": pair_id,
            "trait_i": a,
            "trait_j": b,
            "cosine": cos_ab,
            "stratum": assign_stratum(abs(cos_ab)),
            "regime": regime,
            "both_antisocial": cluster_status == "within_antisocial",
        })
        trajectory_frames.append(df_traj)

        summary_pairs.append({
            "trait_a": a,
            "trait_b": b,
            "status": "ok",
            "alpha": COMPOSITION_ALPHA,
            "cos": round(cos_ab, 4),
            "regime": regime,
            "baseline": {k: round(v, 2) for k, v in base.items()},
            "single_a": {k: round(v, 2) for k, v in single_a.items()},
            "single_b": {k: round(v, 2) for k, v in single_b.items()},
            "steered": {k: round(v, 2) for k, v in steered.items()},
            "delta": {
                "trait_a_joint": round(delta_a_joint, 2),
                "trait_b_joint": round(delta_b_joint, 2),
                "trait_a_single": round(delta_a_single, 2),
                "trait_b_single": round(delta_b_single, 2),
                "composition": round(delta_comp, 2),
                "coherence": round(delta_coh, 2),
            },
            "l17_sanity": {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in sanity.items()},
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
                "polarity_inverted_for_steering": sorted(POLARITY_INVERTED),
                "projection_axis": "raw_direction_no_polarity_flip",
                "n_per_question": N_PER_QUESTION,
                "trajectory_layers": trajectory_layers,
                "trajectory_settings": [list(s) for s in TRAJECTORY_SETTINGS],
                "tau_recipe": TAU_RECIPE,
                "tau_value": None,   # filled at end of run after _calibrate_tau_r2
                "pairs": summary_pairs,
            }, f, indent=2)

    # ---------------------------------------------------------------------------
    # Aggregate trajectory parquet (Phase 2 long-form table)
    # ---------------------------------------------------------------------------
    tau_summary: dict | None = None
    if trajectory_frames:
        agg = pd.concat(trajectory_frames, ignore_index=True)
        meta = pd.DataFrame(pairs_meta)
        agg = agg.merge(meta, on="pair_id", how="left")
        TRAJECTORY_AGG_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(TRAJECTORY_AGG_PARQUET, index=False)
        print(
            f"\nTrajectory dataset: {len(agg):,} rows across "
            f"{agg['pair_id'].nunique()} pairs -> {TRAJECTORY_AGG_PARQUET}"
        )

        # τ R2 calibration over the aggregated dataset.
        print(f"Calibrating τ via {TAU_RECIPE} ...")
        tau_summary = _calibrate_tau_r2(agg, trajectory_layers, COMPOSITION_ALPHA)
        TAU_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        TAU_OUT_PATH.write_text(json.dumps(tau_summary, indent=2))
        print(
            f"  τ = {tau_summary['tau']:.4f}  "
            f"(q{int(TAU_Q*100)}={tau_summary.get('q_value_pre_factor', float('nan')):.4f} × {TAU_FACTOR}, "
            f"n_groups={tau_summary['n_groups']}, "
            f"n_draws={tau_summary['n_draws_total']:,})  -> {TAU_OUT_PATH}"
        )

        # Stamp tau into the summary JSON.
        with open(SUMMARY_OUT_PATH) as f:
            summary_payload = json.load(f)
        summary_payload["tau_value"] = tau_summary["tau"]
        summary_payload["tau_summary_path"] = str(TAU_OUT_PATH)
        with open(SUMMARY_OUT_PATH, "w") as f:
            json.dump(summary_payload, f, indent=2)

    # ---------------------------------------------------------------------------
    # End-of-run table
    # ---------------------------------------------------------------------------
    ok_rows = [r for r in summary_pairs if r["status"] == "ok"]
    print()
    header = (
        f"{'trait_a':<14} {'trait_b':<14} "
        f"{'cos':>6} {'regime':<11} "
        f"{'comp_b':>7} {'comp_s':>7} {'Δ_comp':>7} "
        f"{'Δa_jt':>6} {'Δa_sg':>6} {'Δb_jt':>6} {'Δb_sg':>6} "
        f"{'coh_s':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in ok_rows:
        bs, st, dl = r["baseline"], r["steered"], r["delta"]
        print(
            f"{r['trait_a']:<14} {r['trait_b']:<14} "
            f"{r['cos']:>+6.3f} {r['regime']:<11} "
            f"{bs['composition_mean']:>7.2f} {st['composition_mean']:>7.2f} {dl['composition']:>+7.2f} "
            f"{dl['trait_a_joint']:>+6.2f} {dl['trait_a_single']:>+6.2f} "
            f"{dl['trait_b_joint']:>+6.2f} {dl['trait_b_single']:>+6.2f} "
            f"{st['coherence_mean']:>6.1f}"
        )
    print(f"\nSummary saved to {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
