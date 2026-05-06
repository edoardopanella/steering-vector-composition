"""
Paper-grade plots for the unit-norm α-sweep at L=17.

Reads:
    results/anthropic_repl/alpha_sweep_l17_summary.json   (mandatory)
    results/anthropic_repl/validation_summary_layer17.json (optional, raw E9.7 — for raw vs unit overlay)

Writes (analysis/figures/):
    fig_unit_l17_dose_response_judge.{pdf,png}    2-panel: per-trait Δ_trait + Δ_coh vs α_unit
    fig_unit_l17_dose_response_logprob.{pdf,png}  per-trait logprob shift vs α_unit
    fig_unit_l17_pareto.{pdf,png}                 Δ_trait × |Δ_coh|, α_unit as marker
    fig_unit_l17_judge_vs_logprob_a4.{pdf,png}    judge × logprob scatter at α_unit=4
    fig_unit_vs_raw_l17.{pdf,png}                 paired bars: raw α=2 (E9.7) vs unit α=4 (matches median magnitude)

Style: matplotlib + seaborn; colour = trait identity (husl, 9 hues), linestyle =
origin (solid Anthropic, dashed project), marker = origin (○/▢).

Run locally:
    venv/bin/python scripts/anthropic_repl/plot_alpha_sweep_l17.py
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
SUMMARY_PATH_UNIT = Path("results/anthropic_repl/alpha_sweep_l17_summary.json")
SUMMARY_PATH_RAW = Path("results/anthropic_repl/validation_summary_layer17.json")
FIG_DIR = Path("analysis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_RELEASED = {
    "apathetic", "evil", "hallucinating", "humorous", "impolite", "optimistic", "sycophantic",
}

JUDGE_DELTA_THRESHOLD = 50.0
LOGPROB_THRESHOLD = 0.5

TRAITS_ORDER: list[str] = [
    "apathetic", "evil", "hallucinating", "humorous", "impolite", "sycophantic",
    "confidence", "formality", "power_seeking",
]

_HUSL = sns.color_palette("husl", n_colors=len(TRAITS_ORDER))
TRAIT_COLOR: dict[str, tuple] = {t: _HUSL[i] for i, t in enumerate(TRAITS_ORDER)}
COLOR_THRESHOLD = (0.30, 0.30, 0.30)


def trait_color(t: str) -> tuple:
    return TRAIT_COLOR.get(t, (0.5, 0.5, 0.5))


def trait_linestyle(t: str) -> str:
    return "-" if t in ANTHROPIC_RELEASED else "--"


def trait_marker(t: str) -> str:
    return "o" if t in ANTHROPIC_RELEASED else "s"


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

def load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    with open(path) as f:
        return json.load(f)


def extract_dose_response(summary: dict) -> dict:
    alphas = [float(a) for a in summary["alphas"]]
    out: dict[str, dict] = {}
    for r in summary["traits"]:
        if r["status"] != "ok":
            continue
        bj = r["llm_judge"]
        lp = r["logprob"]
        rows = {}
        for a in alphas:
            j = bj["alphas"][str(a)]
            l = lp["alphas"][str(a)]
            rows[a] = {
                "delta_trait": j["delta_trait"],
                "delta_coh": j["delta_coh"],
                "steer_trait": j["steer_trait"],
                "steer_coh": j["steer_coh"],
                "lp_shift": l["mean_shift"],
                "lp_pass": l["pass_threshold"],
            }
        out[r["trait"]] = {
            "norm": r.get("norm_at_layer17"),
            "base_trait": bj["base_trait"],
            "base_coh": bj["base_coh"],
            "lp_unsteered": lp["mean_unsteered"],
            "alphas": rows,
        }
    return out


# ---------------------------------------------------------------------------
# Legend helper
# ---------------------------------------------------------------------------

def _build_legend_handles(per_trait: dict) -> list:
    trait_handles = [
        plt.Line2D([0], [0], color=trait_color(t), linestyle=trait_linestyle(t),
                   marker=trait_marker(t), linewidth=1.6, markersize=5, label=t)
        for t in TRAITS_ORDER if t in per_trait
    ]
    origin_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.4,
                   label="Anthropic-released (solid)"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.4,
                   label="Project-generated (dashed)"),
    ]
    return trait_handles + origin_handles


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_dose_response_judge(per_trait: dict, alphas: list[float], save_path: Path) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.8), sharex=True)
    xs = [0.0] + alphas

    for trait in TRAITS_ORDER:
        if trait not in per_trait:
            continue
        s = per_trait[trait]
        d_trait = [0.0] + [s["alphas"][a]["delta_trait"] for a in alphas]
        d_coh = [0.0] + [s["alphas"][a]["delta_coh"] for a in alphas]
        c, ls, mk = trait_color(trait), trait_linestyle(trait), trait_marker(trait)
        axL.plot(xs, d_trait, marker=mk, linestyle=ls, color=c, linewidth=1.6, markersize=5)
        axR.plot(xs, d_coh, marker=mk, linestyle=ls, color=c, linewidth=1.6, markersize=5)

    axL.axhline(JUDGE_DELTA_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    axL.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    axR.axhline(0, color="black", linewidth=0.5, alpha=0.4)

    axL.set_xlabel(r"unit-norm steering coefficient $\alpha_{\mathrm{unit}}$")
    axR.set_xlabel(r"unit-norm steering coefficient $\alpha_{\mathrm{unit}}$")
    axL.set_ylabel(r"$\Delta$ trait (LLM-judge, 0–100)")
    axR.set_ylabel(r"$\Delta$ coherence (LLM-judge, 0–100)")
    axL.set_title("(a) Trait expression dose–response @ L=17, unit-norm vectors")
    axR.set_title("(b) Coherence cost dose–response @ L=17, unit-norm vectors")
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
        c, ls, mk = trait_color(trait), trait_linestyle(trait), trait_marker(trait)
        ax.plot(xs, ys, marker=mk, linestyle=ls, color=c, linewidth=1.6, markersize=5)

    ax.axhline(LOGPROB_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    ax.axhline(-LOGPROB_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)

    ax.set_xlabel(r"unit-norm steering coefficient $\alpha_{\mathrm{unit}}$")
    ax.set_ylabel(r"logprob shift (nats), $\log P_\alpha(\mathrm{trait}) - \log P_\alpha(\mathrm{non\!-\!trait})$ minus baseline")
    ax.set_title("Logprob shift dose–response @ L=17, unit-norm vectors (MWE test split)")
    ax.set_xticks(xs)

    handles = _build_legend_handles(per_trait)
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


def plot_pareto(per_trait: dict, alphas: list[float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    marker_for = {2.0: "o", 4.0: "s", 6.0: "D", 8.0: "^"}
    size_for = {2.0: 60, 4.0: 90, 6.0: 130, 8.0: 170}

    for trait in TRAITS_ORDER:
        if trait not in per_trait:
            continue
        s = per_trait[trait]
        xs_pts = [abs(s["alphas"][a]["delta_coh"]) for a in alphas]
        ys_pts = [s["alphas"][a]["delta_trait"] for a in alphas]
        c, ls = trait_color(trait), trait_linestyle(trait)
        ax.plot(xs_pts, ys_pts, linestyle=ls, color=c, alpha=0.7, linewidth=1.2)
        for a in alphas:
            ax.scatter(
                abs(s["alphas"][a]["delta_coh"]), s["alphas"][a]["delta_trait"],
                marker=marker_for[a], s=size_for[a], color=c, edgecolor="black",
                linewidth=0.5, zorder=3,
            )
        # Label at α=4 (median).
        ref_a = 4.0 if 4.0 in alphas else alphas[len(alphas) // 2]
        x_ref = abs(s["alphas"][ref_a]["delta_coh"])
        y_ref = s["alphas"][ref_a]["delta_trait"]
        ax.annotate(trait, (x_ref, y_ref), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=c, fontweight="bold")

    ax.axhline(JUDGE_DELTA_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$|\Delta$ coherence$|$ (cost, 0–100)")
    ax.set_ylabel(r"$\Delta$ trait (gain, 0–100)")
    ax.set_title(r"Trait gain vs coherence cost @ L=17, unit-norm — $\alpha_{\mathrm{unit}}$ ∈ {2,4,6,8}")

    trait_handles = [
        plt.Line2D([0], [0], color=trait_color(t), linestyle=trait_linestyle(t),
                   marker=trait_marker(t), linewidth=1.6, markersize=5, label=t)
        for t in TRAITS_ORDER if t in per_trait
    ]
    alpha_handles = [
        plt.Line2D([0], [0], marker=marker_for[a], color="white",
                   markerfacecolor="gray", markeredgecolor="black",
                   markersize=np.sqrt(size_for[a]), label=f"α={a}")
        for a in alphas
    ]
    origin_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.4, label="Anthropic"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.4, label="Project"),
    ]
    threshold_handle = [
        plt.Line2D([0], [0], color=COLOR_THRESHOLD, linestyle=":", linewidth=1,
                   label=f"Δ_trait = {int(JUDGE_DELTA_THRESHOLD)}")
    ]
    leg1 = ax.legend(handles=trait_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     fontsize=8, title="trait")
    ax.add_artist(leg1)
    ax.legend(handles=alpha_handles + origin_handles + threshold_handle,
              loc="lower right", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


def plot_judge_vs_logprob(per_trait: dict, alpha: float, save_path: Path) -> None:
    """Judge × logprob scatter at one α."""
    from scipy.stats import pearsonr, spearmanr

    xs, ys, names = [], [], []
    for trait, s in per_trait.items():
        a = s["alphas"][alpha]
        xs.append(a["delta_trait"])
        ys.append(a["lp_shift"])
        names.append(trait)
    xa, ya = np.asarray(xs), np.asarray(ys)
    pr, _ = pearsonr(xa, ya)
    sp, _ = spearmanr(xa, ya)
    m, b = np.polyfit(xa, ya, 1)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for x, y, n in zip(xs, ys, names):
        c, mk = trait_color(n), trait_marker(n)
        ax.scatter(x, y, s=110, color=c, edgecolor="black", linewidth=0.6, marker=mk, zorder=3)
        ax.annotate(n, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=c, fontweight="bold")
    xx = np.linspace(min(xa) - 5, max(xa) + 5, 100)
    ax.plot(xx, m * xx + b, color="black", linewidth=0.7, alpha=0.6)
    ax.axhline(LOGPROB_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    ax.axhline(-LOGPROB_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    ax.axvline(JUDGE_DELTA_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xlabel(rf"$\Delta$ trait (LLM-judge, $\alpha_{{\mathrm{{unit}}}}$={alpha})")
    ax.set_ylabel(rf"logprob shift (nats, $\alpha_{{\mathrm{{unit}}}}$={alpha})")
    ax.set_title(rf"LLM-judge × logprob @ L=17, $\alpha_{{\mathrm{{unit}}}}$={alpha} — Pearson r={pr:.2f}, Spearman ρ={sp:.2f}")

    trait_handles = [
        plt.Line2D([0], [0], color=trait_color(t), marker=trait_marker(t),
                   linestyle="None", markersize=8, label=t)
        for t in TRAITS_ORDER if t in per_trait
    ]
    ax.legend(handles=trait_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=8, title="trait")
    fig.tight_layout()
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)


def plot_unit_vs_raw(per_trait_unit: dict, raw_summary: dict, alpha_unit_match: float, save_path: Path) -> None:
    """Paired bars: raw α=2 (E9.7) vs unit α=alpha_unit_match. Bars colored by trait, hatch = raw."""
    raw_by_trait = {r["trait"]: r for r in raw_summary["traits"] if r["status"] == "ok"}
    rows = []
    for trait in TRAITS_ORDER:
        if trait not in per_trait_unit or trait not in raw_by_trait:
            continue
        r_raw = raw_by_trait[trait]
        u = per_trait_unit[trait]["alphas"][alpha_unit_match]
        raw_a2 = r_raw["llm_judge"]["alphas"]["2.0"]
        raw_lp_a2 = r_raw["logprob"]["alphas"]["2.0"]
        rows.append({
            "trait": trait,
            "judge_raw": raw_a2["delta_trait"],
            "judge_unit": u["delta_trait"],
            "lp_raw": raw_lp_a2["mean_shift"],
            "lp_unit": u["lp_shift"],
        })
    if not rows:
        print("[warn] no overlap traits — skipping unit-vs-raw plot")
        return

    rows.sort(key=lambda r: -r["judge_unit"])
    traits = [r["trait"] for r in rows]
    colors = [trait_color(t) for t in traits]
    y = np.arange(len(traits))
    h = 0.4

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, max(4.0, 0.55 * len(traits))), sharey=True)

    axL.barh(y - h / 2, [r["judge_raw"] for r in rows], height=h,
             color=colors, edgecolor="black", linewidth=0.6, hatch="///", alpha=0.85)
    axL.barh(y + h / 2, [r["judge_unit"] for r in rows], height=h,
             color=colors, edgecolor="black", linewidth=0.6)
    axL.axvline(JUDGE_DELTA_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    axL.axvline(0, color="black", linewidth=0.5, alpha=0.4)
    axL.set_yticks(y)
    axL.set_yticklabels(traits)
    for tl, t in zip(axL.get_yticklabels(), traits):
        tl.set_color(trait_color(t)); tl.set_fontweight("bold")
    axL.invert_yaxis()
    axL.set_xlabel(r"$\Delta$ trait (LLM-judge)")
    axL.set_title("(a) LLM-judge")

    axR.barh(y - h / 2, [r["lp_raw"] for r in rows], height=h,
             color=colors, edgecolor="black", linewidth=0.6, hatch="///", alpha=0.85)
    axR.barh(y + h / 2, [r["lp_unit"] for r in rows], height=h,
             color=colors, edgecolor="black", linewidth=0.6)
    axR.axvline(LOGPROB_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    axR.axvline(-LOGPROB_THRESHOLD, color=COLOR_THRESHOLD, linestyle=":", linewidth=0.8)
    axR.axvline(0, color="black", linewidth=0.5, alpha=0.4)
    axR.set_xlabel("logprob shift (nats)")
    axR.set_title("(b) Logprob (MWE test split)")

    layer_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="lightgray", edgecolor="black",
                      hatch="///", label="raw α=2 (E9.7)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="lightgray", edgecolor="black",
                      label=f"unit α={alpha_unit_match}"),
    ]
    fig.legend(handles=layer_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Raw α=2 (E9.7) vs unit α={alpha_unit_match} @ L=17 — bar colour = trait, hatch = raw",
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
    summary_unit = load_summary(SUMMARY_PATH_UNIT)
    alphas = [float(a) for a in summary_unit["alphas"]]
    per_trait = extract_dose_response(summary_unit)
    print(f"Loaded unit α-sweep: {len(per_trait)} traits, alphas={alphas}")

    p1 = FIG_DIR / "fig_unit_l17_dose_response_judge.pdf"
    plot_dose_response_judge(per_trait, alphas, p1); print(f"  wrote {p1}")

    p2 = FIG_DIR / "fig_unit_l17_dose_response_logprob.pdf"
    plot_dose_response_logprob(per_trait, alphas, p2); print(f"  wrote {p2}")

    p3 = FIG_DIR / "fig_unit_l17_pareto.pdf"
    plot_pareto(per_trait, alphas, p3); print(f"  wrote {p3}")

    # Judge × logprob scatter at α=4 (closest to median raw α=2 effective magnitude).
    p4 = FIG_DIR / "fig_unit_l17_judge_vs_logprob_a4.pdf"
    plot_judge_vs_logprob(per_trait, 4.0, p4); print(f"  wrote {p4}")

    if SUMMARY_PATH_RAW.exists():
        raw_summary = load_summary(SUMMARY_PATH_RAW)
        p5 = FIG_DIR / "fig_unit_vs_raw_l17.pdf"
        plot_unit_vs_raw(per_trait, raw_summary, 4.0, p5); print(f"  wrote {p5}")
    else:
        print(f"  [skip] {SUMMARY_PATH_RAW} not found — unit-vs-raw plot skipped")


if __name__ == "__main__":
    main()
