"""
Diagnostic plots for composition-scoring results.

Loads results/composition_scoring_l17_summary.json (per-pair regimes + Δ +
baselines + steered means + cos) and produces a single multi-panel PDF for
sanity-checking the RQ1 Part A dataset BEFORE running any inferential
analysis. The aim is to surface dataset pathologies that would invalidate
downstream stats — class imbalance, Tan unsteerability, regime threshold
fragility, coherence collapse — without committing to an analysis pipeline.

Panels:
  A. Regime distribution            (bar chart, counts per regime)
  B. |cos| vs composition quality Q (scatter, regime-colored; Eq 4 of research_plan.md)
  C. Δ_joint vs Δ_single, axis a    (scatter; on-diagonal = additive ratio 1.0)
  D. Δ_joint vs Δ_single, axis b    (same)
  E. Per-trait mean |Δ_single|      (Tan steerability check — small = unsteerable)
  F. Coherence drop by regime       (boxplot of steered_coh − baseline_coh)

Run:
    python -m scripts.compositions.composition_diagnostics_plot

Output:
    results/figures/composition_diagnostics.pdf
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATH = Path("results/composition_scoring_l17_summary.json")
FIG_OUT = Path("results/figures/composition_diagnostics.pdf")

# Composition-quality stability ε (research_plan.md Eq 4).
Q_EPS = 1.0

# Regime color palette (color-blind-safe, qualitative).
REGIME_COLORS = {
    "additive": "#0072B2",      # blue
    "dominant": "#E69F00",      # orange
    "suppressive": "#CC79A7",   # magenta
    "emergent": "#009E73",      # green
    "mixed": "#999999",         # grey
}
REGIME_ORDER = ["additive", "dominant", "suppressive", "emergent", "mixed"]


def _load_pairs() -> list[dict]:
    s = json.loads(SUMMARY_PATH.read_text())
    return [p for p in s["pairs"] if p.get("status") == "ok"]


def _composition_quality(p: dict) -> float:
    """Q(i,j) = 0.5 * (E_i(1,1)/max(E_i(1,0), ε) + E_j(1,1)/max(E_j(0,1), ε)).

    E_i(1,0) = single_a.trait_a_mean (apathetic-only steering bumps apathetic)
    E_j(0,1) = single_b.trait_b_mean
    E_i(1,1) = steered.trait_a_mean
    E_j(1,1) = steered.trait_b_mean
    """
    e_i_10 = p["single_a"]["trait_a_mean"]
    e_j_01 = p["single_b"]["trait_b_mean"]
    e_i_11 = p["steered"]["trait_a_mean"]
    e_j_11 = p["steered"]["trait_b_mean"]
    return 0.5 * (
        e_i_11 / max(e_i_10, Q_EPS) + e_j_11 / max(e_j_01, Q_EPS)
    )


def _panel_regime_distribution(ax, pairs: list[dict]) -> None:
    counts = Counter(p["regime"] for p in pairs)
    xs = REGIME_ORDER
    ys = [counts.get(r, 0) for r in xs]
    colors = [REGIME_COLORS[r] for r in xs]
    bars = ax.bar(xs, ys, color=colors, edgecolor="black", linewidth=0.5)
    for bar, n in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(n), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"A. Regime distribution (n={len(pairs)})", fontsize=11, loc="left")
    ax.set_ylabel("# pairs")
    ax.set_ylim(0, max(ys) * 1.2 + 1)
    ax.tick_params(axis="x", rotation=30)


def _panel_cos_vs_q(ax, pairs: list[dict]) -> None:
    for r in REGIME_ORDER:
        sub = [p for p in pairs if p["regime"] == r]
        if not sub:
            continue
        xs = [abs(p["cos"]) for p in sub]
        ys = [_composition_quality(p) for p in sub]
        ax.scatter(xs, ys, c=REGIME_COLORS[r], label=f"{r} ({len(sub)})",
                   s=45, edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.axhline(1.0, ls="--", color="grey", lw=0.8)
    ax.axhline(0.8, ls=":", color="grey", lw=0.8)
    ax.set_xlabel("|cos(v_a, v_b)|")
    ax.set_ylabel("composition quality Q(i,j)")
    ax.set_title("B. |cos| vs Q — RQ1 headline preview", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))


def _panel_delta_joint_vs_single(ax, pairs: list[dict], axis: str) -> None:
    """axis='a' uses delta.trait_a_joint vs delta.trait_a_single."""
    key_j = f"trait_{axis}_joint"
    key_s = f"trait_{axis}_single"
    for r in REGIME_ORDER:
        sub = [p for p in pairs if p["regime"] == r]
        if not sub:
            continue
        xs = [p["delta"][key_s] for p in sub]
        ys = [p["delta"][key_j] for p in sub]
        ax.scatter(xs, ys, c=REGIME_COLORS[r], s=40, edgecolor="black",
                   linewidth=0.4, alpha=0.85)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0], -5)
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1], 5)
    ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.8)   # y=x = additive
    ax.plot([lo, hi], [0, 0], ":", color="grey", lw=0.5)
    ax.plot([0, 0], [lo, hi], ":", color="grey", lw=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Δ {axis}, single (1, partner=0)")
    ax.set_ylabel(f"Δ {axis}, joint (1,1)")
    label = "C" if axis == "a" else "D"
    ax.set_title(f"{label}. Δ_joint vs Δ_single, axis {axis}", fontsize=11, loc="left")


def _panel_tan_steerability(ax, pairs: list[dict]) -> None:
    """Per-trait mean |Δ_single| over pairs where trait sits on the active axis.

    For each trait t, average |trait_a_single| over pairs with trait_a==t and
    |trait_b_single| over pairs with trait_b==t.
    """
    per_trait: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        per_trait[p["trait_a"]].append(abs(p["delta"]["trait_a_single"]))
        per_trait[p["trait_b"]].append(abs(p["delta"]["trait_b_single"]))
    traits = sorted(per_trait.keys(), key=lambda t: np.mean(per_trait[t]))
    means = [np.mean(per_trait[t]) for t in traits]
    colors = ["#D55E00" if m < 5 else "#0072B2" for m in means]
    ax.barh(traits, means, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(5, ls=":", color="grey", lw=0.8)
    ax.set_xlabel("mean |Δ| on active axis when steered alone")
    ax.set_title("E. Tan steerability — small bars = unsteerable",
                 fontsize=11, loc="left")
    for i, m in enumerate(means):
        ax.text(m + 0.3, i, f"{m:.1f}", va="center", fontsize=8)


def _panel_coherence_drop(ax, pairs: list[dict]) -> None:
    data, labels, colors = [], [], []
    for r in REGIME_ORDER:
        sub = [p for p in pairs if p["regime"] == r]
        if not sub:
            continue
        drops = [p["steered"]["coherence_mean"] - p["baseline"]["coherence_mean"]
                 for p in sub]
        data.append(drops)
        labels.append(f"{r}\n(n={len(sub)})")
        colors.append(REGIME_COLORS[r])
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55)
    for box, c in zip(bp["boxes"], colors):
        box.set(facecolor=c, alpha=0.7, edgecolor="black", linewidth=0.5)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.2)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("Δ coherence (joint − baseline)")
    ax.set_title("F. Coherence drop by regime", fontsize=11, loc="left")


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise SystemExit(
            f"Summary JSON not found at {SUMMARY_PATH}. Run "
            "composition_aggregate_local.py first."
        )
    pairs = _load_pairs()
    if not pairs:
        raise SystemExit("No pairs with status=ok in summary JSON.")

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    (a, b), (c, d), (e, f) = axes
    _panel_regime_distribution(a, pairs)
    _panel_cos_vs_q(b, pairs)
    _panel_delta_joint_vs_single(c, pairs, "a")
    _panel_delta_joint_vs_single(d, pairs, "b")
    _panel_tan_steerability(e, pairs)
    _panel_coherence_drop(f, pairs)
    fig.suptitle(
        f"Composition diagnostics — L*=17, α=4 — n={len(pairs)} pairs",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(FIG_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_OUT}  ({FIG_OUT.stat().st_size / 1024:.0f} KB)")

    # Console summary alongside the figure.
    regime_counts = Counter(p["regime"] for p in pairs)
    print("\nregime counts:", dict(regime_counts))
    qs = [(p["trait_a"], p["trait_b"], abs(p["cos"]), _composition_quality(p), p["regime"]) for p in pairs]
    print(f"\nQ stats: mean={np.mean([q for *_, q, _ in [(*_q,) for _q in qs]]):.2f}  "
          f"min={min(q for *_, q, _ in [(*_q,) for _q in qs]):.2f}  "
          f"max={max(q for *_, q, _ in [(*_q,) for _q in qs]):.2f}")

    # Tan unsteerability list — traits with mean |Δ_single| < 5.
    per_trait: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        per_trait[p["trait_a"]].append(abs(p["delta"]["trait_a_single"]))
        per_trait[p["trait_b"]].append(abs(p["delta"]["trait_b_single"]))
    unsteerable = [t for t in per_trait if np.mean(per_trait[t]) < 5]
    if unsteerable:
        print(f"\nWARNING — traits with mean |Δ_single| < 5 (Tan-unsteerable): {unsteerable}")


if __name__ == "__main__":
    main()
