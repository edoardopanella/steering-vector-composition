"""
Paper-grade plots for the L=17 α-sweep validation (Phase 9.x).

Reads:
    results/anthropic_repl/validation_summary_layer17.json   (mandatory)
    results/anthropic_repl/validation_summary.json           (optional — E7.8 L=16, for
                                                              the L=16 vs L=17 paired-bar plot)

Writes (analysis/figures/):
    fig_l17_dose_response_judge.pdf   2-panel: per-trait Δ_trait vs α, Δ_coh vs α
    fig_l17_dose_response_logprob.pdf 1-panel: per-trait logprob shift vs α
    fig_l17_pareto.pdf                Δ_trait × |Δ_coh| with α as marker, dashed lines connect the
                                       three α points per trait — picks the trait-α sweet spot
    fig_l17_judge_vs_logprob_a2.pdf   scatter at α=2: Δ_trait (x) × logprob shift (y), trait labels
    fig_l17_l16_vs_l17_a2.pdf         paired bars L=16 vs L=17 at α=2 (judge Δ_trait + logprob shift)

Style: matplotlib + seaborn, 300 DPI PDFs + PNG twins.

Run locally (no GPU, no API):
    venv/bin/python scripts/anthropic_repl/plot_validation_layer17.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUMMARY_PATH_L17 = Path("results/anthropic_repl/validation_summary_layer17.json")
SUMMARY_PATH_L16 = Path("results/anthropic_repl/validation_summary.json")
FIG_DIR = Path("analysis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_RELEASED = {
    "apathetic", "evil", "hallucinating", "humorous", "impolite", "optimistic", "sycophantic",
}

JUDGE_DELTA_THRESHOLD = 50.0
LOGPROB_THRESHOLD = 0.5

# Canonical ordering: Anthropic-released first (alphabetical), project-generated second
# (alphabetical). Used for legend ordering and color assignment so the same trait
# always gets the same colour across every plot.
TRAITS_ORDER: list[str] = [
    # Anthropic-released
    "apathetic", "evil", "hallucinating", "humorous", "impolite", "sycophantic",
    # Project-generated
    "confidence", "formality", "power_seeking",
]

# Per-trait colours — 9 distinct hues from seaborn's `husl` palette so each
# trait gets its own identity colour in every plot. Origin (Anthropic vs project)
# is encoded by linestyle (solid vs dashed), not by colour.
_HUSL_PALETTE = sns.color_palette("husl", n_colors=len(TRAITS_ORDER))
TRAIT_COLOR: dict[str, tuple] = {t: _HUSL_PALETTE[i] for i, t in enumerate(TRAITS_ORDER)}

COLOR_THRESHOLD = (0.30, 0.30, 0.30)


def trait_color(trait: str) -> tuple:
    return TRAIT_COLOR.get(trait, (0.5, 0.5, 0.5))


def trait_linestyle(trait: str) -> str:
    return "-" if trait in ANTHROPIC_RELEASED else "--"


def trait_marker(trait: str) -> str:
    return "o" if trait in ANTHROPIC_RELEASED else "s"


def _setup_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times", "Liberation Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def load_summary_l17() -> dict:
    if not SUMMARY_PATH_L17.exists():
        raise FileNotFoundError(f"Missing {SUMMARY_PATH_L17}. Pull cluster results first.")
    with open(SUMMARY_PATH_L17) as f:
        return json.load(f)


def load_summary_l16_optional() -> dict | None:
    if not SUMMARY_PATH_L16.exists():
        return None
    with open(SUMMARY_PATH_L16) as f:
        return json.load(f)


def extract_dose_response(summary: dict) -> dict:
    """Per-trait dict: {alpha: float -> {delta_trait, delta_coh, lp_shift}}, plus baselines."""
    alphas = [float(a) for a in summary["alphas"]]
    out: dict[str, dict] = {}
    for r in summary["traits"]:
        if r["status"] != "ok":
            continue
        bj = r["llm_judge"]
        bt, bc = bj["base_trait"], bj["base_coh"]
        lp = r["logprob"]
        rows = {}
        for a in alphas:
            jslot = bj["alphas"][str(a)]
            lpslot = lp["alphas"][str(a)]
            rows[a] = {
                "delta_trait": jslot["delta_trait"],
                "delta_coh": jslot["delta_coh"],
                "steer_trait": jslot["steer_trait"],
                "steer_coh": jslot["steer_coh"],
                "lp_shift": lpslot["mean_shift"],
                "lp_pass": lpslot["pass_threshold"],
            }
        out[r["trait"]] = {
            "base_trait": bt,
            "base_coh": bc,
            "lp_unsteered": lp["mean_unsteered"],
            "alphas": rows,
        }
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _build_legend_handles(per_trait: dict) -> list:
    """Trait-colour legend handles in TRAITS_ORDER + linestyle origin legend."""
    trait_handles = [
        plt.Line2D(
            [0], [0], color=trait_color(t),
            linestyle=trait_linestyle(t), marker=trait_marker(t),
            linewidth=1.6, markersize=5, label=t,
        )
        for t in TRAITS_ORDER if t in per_trait
    ]
    origin_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.4,
                   label="Anthropic-released (solid)"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.4,
                   label="Project-generated (dashed)"),
    ]
    return trait_handles + origin_handles


def plot_dose_response_judge(per_trait: dict, alphas: list[float], save_path: Path) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.8), sharex=True)
    xs = [0.0] + alphas

    for trait in TRAITS_ORDER:
        if trait not in per_trait:
            continue
        s = per_trait[trait]
        d_trait = [0.0] + [s["alphas"][a]["delta_trait"] for a in alphas]
        d_coh = [0.0] + [s["alphas"][a]["delta_coh"] for a in alphas]
        c = trait_color(trait)
        ls = trait_linestyle(trait)
        mk = trait_marker(trait)
        axL.plot(xs, d_trait, marker=mk, linestyle=ls, color=c, linewidth=1.6, markersize=5)
        axR.plot(xs, d_coh, marker=mk, linestyle=ls, color=c, linewidth=1.6, markersize=5)

    axL.axhline(JUDGE_DELTA_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    axL.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    axR.axhline(0, color="black", linewidth=0.5, alpha=0.4)

    axL.set_xlabel(r"steering coefficient $\alpha$")
    axR.set_xlabel(r"steering coefficient $\alpha$")
    axL.set_ylabel(r"$\Delta$ trait (LLM-judge, 0–100)")
    axR.set_ylabel(r"$\Delta$ coherence (LLM-judge, 0–100)")
    axL.set_title("(a) Trait expression dose–response @ L=17")
    axR.set_title("(b) Coherence cost dose–response @ L=17")
    axL.set_xticks(xs)
    axR.set_xticks(xs)

    handles = _build_legend_handles(per_trait)
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


def plot_dose_response_logprob(per_trait: dict, alphas: list[float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    xs = [0.0] + alphas

    for trait in TRAITS_ORDER:
        if trait not in per_trait:
            continue
        s = per_trait[trait]
        ys = [0.0] + [s["alphas"][a]["lp_shift"] for a in alphas]
        c = trait_color(trait)
        ls = trait_linestyle(trait)
        mk = trait_marker(trait)
        ax.plot(xs, ys, marker=mk, linestyle=ls, color=c, linewidth=1.6, markersize=5)

    ax.axhline(LOGPROB_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    ax.axhline(-LOGPROB_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)

    ax.set_xlabel(r"steering coefficient $\alpha$")
    ax.set_ylabel(r"logprob shift (nats), $\log P_\alpha(\mathrm{trait}) - \log P_\alpha(\mathrm{non\!-\!trait})$ minus baseline")
    ax.set_title("Logprob shift dose–response @ L=17 (MWE test split)")
    ax.set_xticks(xs)

    handles = _build_legend_handles(per_trait)
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


def plot_pareto(per_trait: dict, alphas: list[float], save_path: Path) -> None:
    """Δ_trait vs |Δ_coh|, three points per trait connected by line. Each trait
    has its own colour; line style (solid/dashed) encodes Anthropic/project origin;
    α is encoded by marker shape (○ α=1, ▢ α=2, ◇ α=3)."""
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    marker_for = {1.0: "o", 2.0: "s", 3.0: "D"}
    size_for = {1.0: 60, 2.0: 90, 3.0: 130}

    for trait in TRAITS_ORDER:
        if trait not in per_trait:
            continue
        s = per_trait[trait]
        xs_pts = [abs(s["alphas"][a]["delta_coh"]) for a in alphas]
        ys_pts = [s["alphas"][a]["delta_trait"] for a in alphas]
        c = trait_color(trait)
        ls = trait_linestyle(trait)
        ax.plot(xs_pts, ys_pts, linestyle=ls, color=c, alpha=0.7, linewidth=1.2)
        for a in alphas:
            ax.scatter(
                abs(s["alphas"][a]["delta_coh"]), s["alphas"][a]["delta_trait"],
                marker=marker_for[a], s=size_for[a], color=c, edgecolor="black",
                linewidth=0.5, zorder=3,
            )
        # Label at α=2.
        x2 = abs(s["alphas"][2.0]["delta_coh"])
        y2 = s["alphas"][2.0]["delta_trait"]
        ax.annotate(trait, (x2, y2), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=c, fontweight="bold")

    ax.axhline(JUDGE_DELTA_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$|\Delta$ coherence$|$ (cost, 0–100)")
    ax.set_ylabel(r"$\Delta$ trait (gain, 0–100)")
    ax.set_title("Trait gain vs coherence cost @ L=17 — α∈{1,2,3}")

    # Legend: trait colours/styles, α markers, threshold.
    trait_handles = [
        plt.Line2D([0], [0], color=trait_color(t),
                   linestyle=trait_linestyle(t), marker=trait_marker(t),
                   linewidth=1.6, markersize=5, label=t)
        for t in TRAITS_ORDER if t in per_trait
    ]
    alpha_handles = [
        plt.Line2D([0], [0], marker=marker_for[a], color="white",
                   markerfacecolor="gray", markeredgecolor="black",
                   markersize=np.sqrt(size_for[a]), label=f"α={a}")
        for a in alphas
    ]
    origin_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.4,
                   label="Anthropic (solid)"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.4,
                   label="Project (dashed)"),
    ]
    threshold_handle = [
        plt.Line2D([0], [0], color=COLOR_THRESHOLD, linestyle=":", linewidth=1,
                   label=f"Δ_trait = {int(JUDGE_DELTA_THRESHOLD)}")
    ]
    leg1 = ax.legend(handles=trait_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     fontsize=8, title="trait")
    ax.add_artist(leg1)
    ax.legend(handles=alpha_handles + origin_handles + threshold_handle,
              loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


def plot_judge_vs_logprob_a2(per_trait: dict, save_path: Path) -> None:
    """Scatter Δ_trait (x) vs logprob shift (y) at α=2."""
    from scipy.stats import pearsonr, spearmanr  # local import — only needed here

    xs, ys, names, cols = [], [], [], []
    for trait, s in per_trait.items():
        a = s["alphas"][2.0]
        xs.append(a["delta_trait"])
        ys.append(a["lp_shift"])
        names.append(trait)
        cols.append(trait_color(trait))

    xs_a, ys_a = np.asarray(xs), np.asarray(ys)
    pr, _ = pearsonr(xs_a, ys_a)
    sp, _ = spearmanr(xs_a, ys_a)
    m, b = np.polyfit(xs_a, ys_a, 1)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for x, y, n in zip(xs, ys, names):
        c = trait_color(n)
        mk = trait_marker(n)
        ax.scatter(x, y, s=110, color=c, edgecolor="black", linewidth=0.6,
                   marker=mk, zorder=3)
        ax.annotate(n, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=c, fontweight="bold")
    xx = np.linspace(min(xs_a) - 5, max(xs_a) + 5, 100)
    ax.plot(xx, m * xx + b, color="black", linewidth=0.7, alpha=0.6,
            label=f"OLS: y = {m:.3f}x + {b:.2f}")
    ax.axhline(LOGPROB_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    ax.axhline(-LOGPROB_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    ax.axvline(JUDGE_DELTA_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xlabel(r"$\Delta$ trait (LLM-judge, α=2)")
    ax.set_ylabel("logprob shift (nats, α=2)")
    ax.set_title(f"LLM-judge × logprob @ L=17, α=2  —  Pearson r={pr:.2f}, Spearman ρ={sp:.2f}")

    trait_handles = [
        plt.Line2D([0], [0], color=trait_color(t), marker=trait_marker(t),
                   linestyle="None", markersize=8, label=t)
        for t in TRAITS_ORDER if t in per_trait
    ]
    origin_handles = [
        plt.Line2D([0], [0], marker="o", color="black", linestyle="None",
                   markersize=7, label="Anthropic (○)"),
        plt.Line2D([0], [0], marker="s", color="black", linestyle="None",
                   markersize=7, label="Project (▢)"),
    ]
    leg1 = ax.legend(handles=trait_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     fontsize=8, title="trait")
    ax.add_artist(leg1)
    ax.legend(handles=origin_handles + [plt.Line2D([0], [0], color="black", alpha=0.6,
                                                    label=f"OLS y={m:.3f}x+{b:.2f}")],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


def plot_l16_vs_l17_a2(per_trait_l17: dict, summary_l16: dict, save_path: Path) -> None:
    """Paired bars at α=2: Δ_trait and logprob shift, L=16 (E7.8) vs L=17."""
    rows = []
    l16_by_trait = {r["trait"]: r for r in summary_l16["traits"] if r["status"] == "ok"}
    for trait, s17 in per_trait_l17.items():
        if trait not in l16_by_trait:
            continue
        r16 = l16_by_trait[trait]
        # E7.8 schema: r16["llm_judge"]["delta_trait"] (single α=2 by design),
        # r16["logprob"]["mean_shift"].
        rows.append({
            "trait": trait,
            "judge_l16": r16["llm_judge"]["delta_trait"],
            "judge_l17": s17["alphas"][2.0]["delta_trait"],
            "lp_l16": r16["logprob"]["mean_shift"] if r16.get("logprob") else float("nan"),
            "lp_l17": s17["alphas"][2.0]["lp_shift"],
        })

    if not rows:
        print("[warn] no overlap between L=16 and L=17 traits — skipping comparison plot")
        return

    rows.sort(key=lambda r: -r["judge_l17"])
    traits = [r["trait"] for r in rows]
    colors = [trait_color(t) for t in traits]
    y = np.arange(len(traits))
    h = 0.4

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, max(4.0, 0.55 * len(traits))), sharey=True)

    # L=16 → hatched bar, L=17 → solid bar; both use the trait colour.
    axL.barh(y - h / 2, [r["judge_l16"] for r in rows], height=h,
             color=colors, edgecolor="black", linewidth=0.6, hatch="///", alpha=0.85)
    axL.barh(y + h / 2, [r["judge_l17"] for r in rows], height=h,
             color=colors, edgecolor="black", linewidth=0.6)
    axL.axvline(JUDGE_DELTA_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    axL.axvline(0, color="black", linewidth=0.5, alpha=0.4)
    axL.set_yticks(y)
    axL.set_yticklabels(traits)
    for tick_label, t in zip(axL.get_yticklabels(), traits):
        tick_label.set_color(trait_color(t))
        tick_label.set_fontweight("bold")
    axL.invert_yaxis()
    axL.set_xlabel(r"$\Delta$ trait (LLM-judge, α=2)")
    axL.set_title("(a) LLM-judge")

    axR.barh(y - h / 2, [r["lp_l16"] for r in rows], height=h,
             color=colors, edgecolor="black", linewidth=0.6, hatch="///", alpha=0.85)
    axR.barh(y + h / 2, [r["lp_l17"] for r in rows], height=h,
             color=colors, edgecolor="black", linewidth=0.6)
    axR.axvline(LOGPROB_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    axR.axvline(-LOGPROB_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    axR.axvline(0, color="black", linewidth=0.5, alpha=0.4)
    axR.set_xlabel("logprob shift (nats, α=2)")
    axR.set_title("(b) Logprob (MWE test split)")

    layer_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="lightgray", edgecolor="black",
                      hatch="///", label="L=16 (E7.8)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="lightgray", edgecolor="black",
                      label="L=17"),
    ]
    fig.legend(handles=layer_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("L=16 (E7.8) vs L=17 at α=2  —  bar colour = trait identity, hatch = L=16",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_style()
    summary17 = load_summary_l17()
    alphas = [float(a) for a in summary17["alphas"]]
    per_trait = extract_dose_response(summary17)

    print(f"Loaded L=17 summary: {len(per_trait)} traits, alphas={alphas}")

    # 1. Dose-response — judge (Δ_trait + Δ_coh).
    p1 = FIG_DIR / "fig_l17_dose_response_judge.pdf"
    plot_dose_response_judge(per_trait, alphas, p1)
    print(f"  wrote {p1}")

    # 2. Dose-response — logprob.
    p2 = FIG_DIR / "fig_l17_dose_response_logprob.pdf"
    plot_dose_response_logprob(per_trait, alphas, p2)
    print(f"  wrote {p2}")

    # 3. Pareto — Δ_trait × |Δ_coh|.
    p3 = FIG_DIR / "fig_l17_pareto.pdf"
    plot_pareto(per_trait, alphas, p3)
    print(f"  wrote {p3}")

    # 4. Judge × logprob scatter at α=2.
    p4 = FIG_DIR / "fig_l17_judge_vs_logprob_a2.pdf"
    plot_judge_vs_logprob_a2(per_trait, p4)
    print(f"  wrote {p4}")

    # 5. L=16 vs L=17 at α=2 (only if E7.8 summary present).
    summary16 = load_summary_l16_optional()
    if summary16 is not None:
        p5 = FIG_DIR / "fig_l17_l16_vs_l17_a2.pdf"
        plot_l16_vs_l17_a2(per_trait, summary16, p5)
        print(f"  wrote {p5}")
    else:
        print(f"  [skip] {SUMMARY_PATH_L16} not found — comparison plot skipped")


if __name__ == "__main__":
    main()
