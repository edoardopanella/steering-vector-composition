"""
RQ2 Phase 3 headline trajectory figure.

For one additive pair and one non-additive pair, plot the projection
trajectories π_a(L) and π_b(L) under single-vector steering vs joint steering
across downstream layers L ∈ [L*, L_final]. The figure shows at a glance
whether the joint trajectory tracks the single trajectory (additive case) or
diverges (non-additive case).

Layout (2 rows × 2 cols):
                    axis a (π_a)                     axis b (π_b)
  additive pair    π_a^(1,0) vs π_a^(1,1)         π_b^(0,1) vs π_b^(1,1)
  non-additive     π_a^(1,0) vs π_a^(1,1)         π_b^(0,1) vs π_b^(1,1)

Each curve is mean over 100 prompts (20 questions × 5 completions) per layer,
±1 SEM band. Auto-picks the visually strongest representative from each class
unless ADDITIVE_PAIR / NON_ADDITIVE_PAIR overrides are set.

Run:
    python -m scripts.compositions.composition_headline_trajectory_plot

Output:
    results/figures/composition_headline_trajectory.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AGG_PARQUET = Path("results/composition/v1_phase12_normFalse_a4/trajectories/aggregate.parquet")
SUMMARY_PATH = Path("results/composition/v1_phase12_normFalse_a4/scoring/summary.json")
FIG_OUT = Path("results/figures/composition_headline_trajectory.pdf")

ALPHA = 4.0

# Manual pair overrides. Leave as None for auto-pick.
ADDITIVE_PAIR: tuple[str, str] | None = None
NON_ADDITIVE_PAIR: tuple[str, str] | None = None
# When auto-picking the non-additive representative, restrict to these regimes:
NON_ADDITIVE_REGIMES = {"suppressive", "emergent"}

# Colour scheme:
COLOR_SINGLE = "#0072B2"   # blue
COLOR_JOINT = "#D55E00"    # vermillion
BAND_ALPHA = 0.18


def _load_summary_pairs() -> list[dict]:
    s = json.loads(SUMMARY_PATH.read_text())
    return [p for p in s["pairs"] if p.get("status") == "ok"]


def _pick_additive(pairs: list[dict]) -> dict:
    if ADDITIVE_PAIR is not None:
        a, b = sorted(ADDITIVE_PAIR)
        for p in pairs:
            if (p["trait_a"], p["trait_b"]) == (a, b):
                return p
        raise SystemExit(f"manual override pair {ADDITIVE_PAIR} not found")
    add = [p for p in pairs if p["regime"] == "additive"]
    if not add:
        raise SystemExit("no additive pairs in summary")
    # pick the one where BOTH single-axis Δ are biggest (avoid Tan-borderline cases)
    add.sort(key=lambda p: min(abs(p["delta"]["trait_a_single"]),
                               abs(p["delta"]["trait_b_single"])), reverse=True)
    return add[0]


def _pick_non_additive(pairs: list[dict]) -> dict:
    if NON_ADDITIVE_PAIR is not None:
        a, b = sorted(NON_ADDITIVE_PAIR)
        for p in pairs:
            if (p["trait_a"], p["trait_b"]) == (a, b):
                return p
        raise SystemExit(f"manual override pair {NON_ADDITIVE_PAIR} not found")
    cands = [p for p in pairs if p["regime"] in NON_ADDITIVE_REGIMES]
    if not cands:
        raise SystemExit("no suppressive/emergent pairs in summary")
    # pick the one with biggest joint-vs-single divergence magnitude (sum across axes)
    def score(p: dict) -> float:
        da = abs(p["delta"]["trait_a_joint"] - p["delta"]["trait_a_single"])
        db = abs(p["delta"]["trait_b_joint"] - p["delta"]["trait_b_single"])
        return da + db
    cands.sort(key=score, reverse=True)
    return cands[0]


def _per_layer_stats(
    agg: pd.DataFrame,
    pair_id: int,
    behaviour_index: int,
    alpha_i: float,
    alpha_j: float,
) -> pd.DataFrame:
    """Return per-layer mean + SEM of projection_value across prompts for one
    (pair, axis, setting) combo."""
    sub = agg[
        (agg.pair_id == pair_id)
        & (agg.behaviour_index == behaviour_index)
        & (agg.alpha_i == alpha_i)
        & (agg.alpha_j == alpha_j)
    ]
    stats = (
        sub.groupby("layer")["projection_value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["sem"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
    return stats.sort_values("layer").reset_index(drop=True)


def _draw_axis_panel(
    ax,
    agg: pd.DataFrame,
    pair: dict,
    axis: int,
    title_prefix: str,
) -> None:
    """One panel: (single) vs (joint) trajectory + ±1 SEM bands."""
    pair_id = next(
        i for i, p in enumerate(_load_summary_pairs()) if (p["trait_a"], p["trait_b"]) == (pair["trait_a"], pair["trait_b"])
    )
    pair_id_in_agg = int(agg[(agg.trait_i == pair["trait_a"]) & (agg.trait_j == pair["trait_b"])]["pair_id"].iloc[0])

    if axis == 0:
        single_label = "single (1, 0): π_a"
        ai_s, aj_s = ALPHA, 0.0
        trait_name = pair["trait_a"]
    else:
        single_label = "single (0, 1): π_b"
        ai_s, aj_s = 0.0, ALPHA
        trait_name = pair["trait_b"]

    single = _per_layer_stats(agg, pair_id_in_agg, axis, ai_s, aj_s)
    joint = _per_layer_stats(agg, pair_id_in_agg, axis, ALPHA, ALPHA)

    ax.plot(single["layer"], single["mean"], color=COLOR_SINGLE, lw=2,
            label=single_label)
    ax.fill_between(single["layer"], single["mean"] - single["sem"],
                    single["mean"] + single["sem"], color=COLOR_SINGLE, alpha=BAND_ALPHA)
    ax.plot(joint["layer"], joint["mean"], color=COLOR_JOINT, lw=2, ls="--",
            label=f"joint (1, 1): π_{'a' if axis == 0 else 'b'}")
    ax.fill_between(joint["layer"], joint["mean"] - joint["sem"],
                    joint["mean"] + joint["sem"], color=COLOR_JOINT, alpha=BAND_ALPHA)
    ax.axvline(17, color="grey", ls=":", lw=0.7)
    ax.text(17.1, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.0,
            "L*", fontsize=8, color="grey", va="top")
    ax.set_xlabel("layer L")
    ax.set_ylabel(f"π_{'a' if axis == 0 else 'b'}(L) — projection on v_{trait_name}^(L*)")
    ax.set_title(
        f"{title_prefix} — {pair['trait_a']} + {pair['trait_b']}  "
        f"(regime={pair['regime']}, cos={pair['cos']:+.3f})",
        fontsize=10, loc="left",
    )
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)


def main() -> None:
    if not AGG_PARQUET.exists():
        raise SystemExit(f"missing {AGG_PARQUET}: run composition_aggregate_local.py first")
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"missing {SUMMARY_PATH}")
    agg = pd.read_parquet(AGG_PARQUET)
    pairs = _load_summary_pairs()

    additive = _pick_additive(pairs)
    nonadd = _pick_non_additive(pairs)
    print(f"additive pair    : {additive['trait_a']} + {additive['trait_b']}  "
          f"(regime={additive['regime']}, cos={additive['cos']:+.3f})")
    print(f"  Δa_single={additive['delta']['trait_a_single']:+.2f}  Δb_single={additive['delta']['trait_b_single']:+.2f}")
    print(f"  Δa_joint ={additive['delta']['trait_a_joint']:+.2f}  Δb_joint ={additive['delta']['trait_b_joint']:+.2f}")
    print(f"non-additive pair: {nonadd['trait_a']} + {nonadd['trait_b']}  "
          f"(regime={nonadd['regime']}, cos={nonadd['cos']:+.3f})")
    print(f"  Δa_single={nonadd['delta']['trait_a_single']:+.2f}  Δb_single={nonadd['delta']['trait_b_single']:+.2f}")
    print(f"  Δa_joint ={nonadd['delta']['trait_a_joint']:+.2f}  Δb_joint ={nonadd['delta']['trait_b_joint']:+.2f}")

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    _draw_axis_panel(axes[0, 0], agg, additive, axis=0, title_prefix="A. Additive — axis a")
    _draw_axis_panel(axes[0, 1], agg, additive, axis=1, title_prefix="B. Additive — axis b")
    _draw_axis_panel(axes[1, 0], agg, nonadd,   axis=0, title_prefix="C. Non-additive — axis a")
    _draw_axis_panel(axes[1, 1], agg, nonadd,   axis=1, title_prefix="D. Non-additive — axis b")
    fig.suptitle(
        "RQ2 Phase 3 headline — π trajectories under single vs joint steering",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(FIG_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {FIG_OUT}  ({FIG_OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
