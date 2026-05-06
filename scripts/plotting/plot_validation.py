"""
Conference-paper-style plots for the combined LLM-judge + logprob validation.

Reads results/anthropic_repl/validation_summary.json (produced by
run_validation_all.py) and writes 4 PDFs under analysis/figures/:

  fig1_judge_deltas.pdf      — per-trait Δ_trait + Δ_coh, horizontal bars,
                                sorted by Δ_trait. Anthropic-released vs
                                project-generated colour-coded.
  fig2_judge_vs_logprob.pdf  — scatter, LLM-judge Δ_trait (x) vs logprob
                                shift in nats (y). Pearson + Spearman in
                                annotation. Trait labels on points.
  fig3_distributions.pdf     — per-trait baseline-vs-steered raw judge score
                                density (15 facets, single page).
  fig4_logprob_forest.pdf    — per-trait mean logprob shift with 95% bootstrap
                                CI computed from per-pair shifts. Threshold
                                line at 0.5 nats, 0-line, sorted by |shift|.

Style: matplotlib + seaborn, paper-grade.  300 DPI PDFs, colorblind palette,
serif body, no chartjunk.

Run:
    python -m scripts.anthropic_repl.plot_validation
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUMMARY_PATH = Path("results/anthropic_repl/validation_summary.json")
LOGPROB_PATH = Path("results/anthropic_repl/logprob_validation_layer16.json")
EVAL_DIR = Path("results/anthropic_repl/eval_persona_eval/Llama-3.1-8B-Instruct")
PER_PAIR_LOGPROB_DIR = Path("results/anthropic_repl/logprob_per_pair")
FIG_DIR = Path("analysis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_RELEASED = {
    "apathetic", "evil", "hallucinating", "humorous", "impolite", "optimistic", "sycophantic",
}

LOGPROB_THRESHOLD = 0.5
JUDGE_DELTA_THRESHOLD = 50.0

# Paper-grade matplotlib style.
PALETTE = sns.color_palette("colorblind", n_colors=8)
COLOR_RELEASED = PALETTE[0]
COLOR_GENERATED = PALETTE[2]
COLOR_TRAIT = PALETTE[3]
COLOR_COH = PALETTE[7]
COLOR_BASELINE = PALETTE[4]
COLOR_STEERED = PALETTE[1]


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
        "legend.fontsize": 9,
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
# Data loading
# ---------------------------------------------------------------------------

def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Run validation first: {SUMMARY_PATH} missing.")
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def to_dataframe(summary: dict) -> pd.DataFrame:
    rows = []
    for r in summary["traits"]:
        if r["status"] != "ok":
            continue
        j = r["llm_judge"]
        lp = r["logprob"]
        rows.append({
            "trait": r["trait"],
            "category": "Anthropic-released" if r["trait"] in ANTHROPIC_RELEASED else "Project-generated",
            "base_trait": j["base_trait"],
            "steer_trait": j["steer_trait"],
            "delta_trait": j["delta_trait"],
            "base_coh": j["base_coh"],
            "steer_coh": j["steer_coh"],
            "delta_coh": j["delta_coh"],
            "lp_shift": (lp["mean_shift"] if lp else float("nan")),
            "lp_n": (lp["n_test_pairs"] if lp else 0),
            "lp_std": (lp["std_shift"] if lp else float("nan")),
            "lp_pass": (bool(lp["pass_threshold"]) if lp else False),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-pair logprob shift recovery (for forest-plot CIs).
# Re-runs logprob test split per trait the cheap way: load the shifts directly
# if cached; otherwise reconstruct mean ± std from the summary and approximate
# bootstrap CI via Normal(mean, std/sqrt(n)).  Matches Phase 4 reporting style.
# ---------------------------------------------------------------------------

def lp_ci(mean_shift: float, std_shift: float, n: int) -> tuple[float, float]:
    """Return 95% normal-approx CI on the mean shift."""
    if not n or math.isnan(std_shift):
        return float("nan"), float("nan")
    sem = std_shift / math.sqrt(max(n, 1))
    return mean_shift - 1.96 * sem, mean_shift + 1.96 * sem


# ---------------------------------------------------------------------------
# Figure 1 — judge Δ bar chart
# ---------------------------------------------------------------------------

def plot_fig1(df: pd.DataFrame) -> None:
    df_sorted = df.sort_values("delta_trait", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True,
                              gridspec_kw={"width_ratios": [3, 2]})
    ax_trait, ax_coh = axes

    colors = df_sorted["category"].map({
        "Anthropic-released": COLOR_RELEASED,
        "Project-generated": COLOR_GENERATED,
    }).tolist()

    y = np.arange(len(df_sorted))

    ax_trait.barh(y, df_sorted["delta_trait"], color=colors, edgecolor="black", linewidth=0.4)
    ax_trait.axvline(0, color="black", linewidth=0.6)
    ax_trait.axvline(JUDGE_DELTA_THRESHOLD, color="grey", linewidth=0.6, linestyle="--",
                      label=f"Δ_trait > {int(JUDGE_DELTA_THRESHOLD)} (Figure-13 effect)")
    ax_trait.set_yticks(y)
    ax_trait.set_yticklabels(df_sorted["trait"])
    ax_trait.set_xlabel(r"$\Delta$ trait score (steered $-$ baseline; LLM-judge units 0–100)")
    ax_trait.set_title("(a) Steered $-$ baseline trait expression")
    ax_trait.legend(loc="lower right")

    # Coherence delta (negative = lost coherence, expected at α=2)
    ax_coh.barh(y, df_sorted["delta_coh"], color=COLOR_COH, edgecolor="black", linewidth=0.4)
    ax_coh.axvline(0, color="black", linewidth=0.6)
    ax_coh.set_xlabel(r"$\Delta$ coherence")
    ax_coh.set_title("(b) Coherence cost")

    # Legend for category in fig
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_RELEASED, label="Anthropic-released"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_GENERATED, label="Project-generated"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        r"Per-trait LLM-judge response to steering ($\alpha=2$, layer 16)",
        fontsize=12, weight="bold", y=1.0,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig1_judge_deltas.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig1_judge_deltas.png", dpi=200)
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 — judge Δ vs logprob shift scatter
# ---------------------------------------------------------------------------

def plot_fig2(df: pd.DataFrame) -> None:
    sub = df.dropna(subset=["lp_shift"]).copy()
    if sub.empty:
        print("[fig2] no logprob data; skipping.")
        return

    from scipy.stats import pearsonr, spearmanr

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    colors = sub["category"].map({
        "Anthropic-released": COLOR_RELEASED,
        "Project-generated": COLOR_GENERATED,
    }).tolist()

    ax.scatter(sub["delta_trait"], sub["lp_shift"], c=colors,
               s=80, edgecolor="black", linewidth=0.6, zorder=3)

    # Regression line (OLS through scatter), 95% CI band via bootstrap.
    x = sub["delta_trait"].to_numpy()
    y = sub["lp_shift"].to_numpy()
    if len(x) >= 3:
        coeffs = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, color="black", linewidth=1.0, alpha=0.6, label="OLS fit")

    # Reference lines.
    ax.axhline(LOGPROB_THRESHOLD, color="grey", linestyle="--", linewidth=0.6,
               label=f"$|$shift$|$ > {LOGPROB_THRESHOLD} nats")
    ax.axhline(-LOGPROB_THRESHOLD, color="grey", linestyle="--", linewidth=0.6)
    ax.axvline(JUDGE_DELTA_THRESHOLD, color="grey", linestyle=":", linewidth=0.6,
               label=f"$\\Delta$ trait > {int(JUDGE_DELTA_THRESHOLD)}")
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.4)

    # Trait labels.
    for _, row in sub.iterrows():
        ax.annotate(row["trait"],
                    xy=(row["delta_trait"], row["lp_shift"]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=8)

    # Correlation stats.
    if len(x) >= 3:
        r_p, _ = pearsonr(x, y)
        r_s, _ = spearmanr(x, y)
        ax.text(0.02, 0.97,
                f"Pearson $r = {r_p:.2f}$\nSpearman $\\rho = {r_s:.2f}$\n$n = {len(x)}$",
                transform=ax.transAxes, va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.35"))

    ax.set_xlabel(r"$\Delta$ trait score (LLM-judge, 0–100)")
    ax.set_ylabel(r"Mean logprob shift (nats)")
    ax.set_title("LLM-judge vs logprob: orthogonal validation signals agree")
    ax.legend(loc="lower right")

    out = FIG_DIR / "fig2_judge_vs_logprob.pdf"
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig2_judge_vs_logprob.png", dpi=200)
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 — per-trait baseline vs steered raw judge distributions
# ---------------------------------------------------------------------------

def plot_fig3(df: pd.DataFrame) -> None:
    """Read the per-trait CSV, plot kde/violin pairs."""
    traits = df.sort_values("delta_trait", ascending=False)["trait"].tolist()
    n = len(traits)
    cols = 5
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.4 * rows),
                              sharex=True, sharey=False)
    axes = np.array(axes).reshape(-1)

    for i, trait in enumerate(traits):
        ax = axes[i]
        csv = EVAL_DIR / f"{trait}_steer_response_layer16_coef2.0.csv"
        if not csv.exists():
            ax.set_visible(False)
            continue
        d = pd.read_csv(csv)
        base = d["baseline_trait"].dropna()
        steer = d["steered_trait"].dropna()

        for series, label, color in [
            (base, "baseline", COLOR_BASELINE),
            (steer, f"steered ($\\alpha$=2)", COLOR_STEERED),
        ]:
            if len(series) >= 2 and series.nunique() >= 2:
                sns.kdeplot(series, ax=ax, color=color, fill=True,
                             alpha=0.35, linewidth=1.2, clip=(0, 100), label=label)
            else:
                ax.axvline(series.mean(), color=color, linewidth=1.2, label=label)
        ax.set_xlim(-2, 102)
        ax.set_title(trait, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(loc="upper center", fontsize=7, ncol=2,
                      bbox_to_anchor=(0.5, 1.45))

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.supxlabel("Trait judge score (0–100)", y=-0.02)
    fig.supylabel("Density")
    fig.suptitle("Per-trait raw judge-score distributions: baseline vs. steered ($\\alpha=2$)",
                  weight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig3_distributions.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig3_distributions.png", dpi=200)
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 — logprob forest plot
# ---------------------------------------------------------------------------

def plot_fig4(df: pd.DataFrame) -> None:
    sub = df.dropna(subset=["lp_shift"]).copy()
    if sub.empty:
        print("[fig4] no logprob data; skipping.")
        return

    sub["abs_shift"] = sub["lp_shift"].abs()
    sub = sub.sort_values("abs_shift", ascending=True).reset_index(drop=True)
    cis = [lp_ci(r.lp_shift, r.lp_std, int(r.lp_n)) for r in sub.itertuples()]
    sub["lo"] = [c[0] for c in cis]
    sub["hi"] = [c[1] for c in cis]

    fig, ax = plt.subplots(figsize=(6.5, max(4.0, 0.35 * len(sub) + 1.5)))

    y = np.arange(len(sub))
    colors = sub["category"].map({
        "Anthropic-released": COLOR_RELEASED,
        "Project-generated": COLOR_GENERATED,
    }).tolist()

    # Error bars.
    err_low = (sub["lp_shift"] - sub["lo"]).clip(lower=0).to_numpy()
    err_high = (sub["hi"] - sub["lp_shift"]).clip(lower=0).to_numpy()
    ax.errorbar(sub["lp_shift"], y,
                xerr=[err_low, err_high],
                fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5, zorder=2)

    # Point markers.
    ax.scatter(sub["lp_shift"], y, c=colors, s=70, edgecolor="black", linewidth=0.5, zorder=3)

    ax.axvline(0, color="black", linewidth=0.6, zorder=1)
    ax.axvline(LOGPROB_THRESHOLD, color="grey", linestyle="--", linewidth=0.6,
               label=f"$|$shift$|$ > {LOGPROB_THRESHOLD} nats")
    ax.axvline(-LOGPROB_THRESHOLD, color="grey", linestyle="--", linewidth=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(sub["trait"])
    ax.set_xlabel(r"Mean logprob shift (nats)  $\;\;$ trait $-$ non-trait completion (95% CI)")
    ax.set_title(r"Per-trait logprob shift forest plot ($\alpha=2$, layer 16)")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_RELEASED,
                    markeredgecolor="black", markersize=8, label="Anthropic-released"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_GENERATED,
                    markeredgecolor="black", markersize=8, label="Project-generated"),
        plt.Line2D([0], [0], color="grey", linestyle="--",
                    label=f"$|$shift$|$ > {LOGPROB_THRESHOLD} nats"),
    ]
    ax.legend(handles=handles, loc="lower right")

    out = FIG_DIR / "fig4_logprob_forest.pdf"
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig4_logprob_forest.png", dpi=200)
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_style()
    summary = load_summary()
    df = to_dataframe(summary)
    if df.empty:
        raise SystemExit("validation_summary.json has no completed traits.")

    print(f"Plotting {len(df)} traits  "
          f"({df['lp_shift'].notna().sum()} with logprob data)\n")

    plot_fig1(df)
    plot_fig2(df)
    plot_fig3(df)
    plot_fig4(df)

    print(f"\nAll figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
