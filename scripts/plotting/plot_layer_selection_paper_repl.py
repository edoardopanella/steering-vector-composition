"""
Per-layer trait-expression and coherence curves for the paper-faithful layer
selection sweep (job 488481).

Reads:
    results/layer_selection_paper_repl.json

Writes (results/figures/):
    fig_layer_selection_paper_repl.{pdf,png}    3-panel: one per paper trait

Each panel plots trait expression and coherence vs hidden_layer (1..32) for one
trait. Annotated reference lines:
    - vertical line at L=16: the paper's pick for all three traits (§B.4)
    - star marker at our paper-literal argmax L*: argmax steer_trait with no
      coherence filter (which lands at L=18, 15, 32 for evil/syco/halluc)
    - star marker at our coh>=50 argmax: the argmax inside the coherent regime
      — the implicit-cutoff reading of §B.4 that matches the paper
    - shaded "coherence floor failed" region where steer_coh < 50, used to
      visualise the gibberish-wins artefact at very late layers under α=2.5
      for evil/syco and at L=32 for hallucinating (coh = 0.01)

Run locally (no GPU, JSON-only):
    venv/bin/python scripts/plotting/plot_layer_selection_paper_repl.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

SUMMARY_PATH = Path("results/layer_selection_paper_repl.json")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAPER_PICK = 16
COH_FLOOR = 50.0

# Husl palette to keep colour identity consistent with the rest of the report.
# The three paper traits sit at the same hues used in plot_alpha_sweep_l17.py.
TRAITS_ORDER = [
    "apathetic", "evil", "hallucinating", "humorous", "impolite", "sycophantic",
    "confidence", "formality", "power_seeking",
]
_HUSL = sns.color_palette("husl", n_colors=len(TRAITS_ORDER))
TRAIT_COLOR = {t: _HUSL[i] for i, t in enumerate(TRAITS_ORDER)}


def load() -> dict:
    with SUMMARY_PATH.open() as f:
        return json.load(f)


def coh_aware_argmax(layers: dict[int, dict], floor: float) -> int | None:
    eligible = {L: v for L, v in layers.items() if v["steer_coh"] >= floor}
    if not eligible:
        return None
    return max(eligible, key=lambda L: eligible[L]["steer_trait"])


def plot_one(ax, trait: str, s: dict) -> None:
    coef = s["coef"]
    layers = {int(L): v for L, v in s["layers"].items()}
    xs = sorted(layers)
    trait_y = [layers[L]["steer_trait"] for L in xs]
    coh_y = [layers[L]["steer_coh"] for L in xs]
    colour = TRAIT_COLOR[trait]

    # Trait line (left y-axis).
    ax.plot(xs, trait_y, color=colour, marker="o", markersize=4, linewidth=1.8,
            label=f"trait expression (α={coef})")
    ax.set_xlim(0.5, 32.5)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("hidden layer (1-indexed)")
    ax.set_ylabel("trait expression", color=colour)
    ax.tick_params(axis="y", labelcolor=colour)
    ax.set_xticks(list(range(1, 33, 2)))
    ax.grid(alpha=0.25)

    # Coherence line (right y-axis).
    ax2 = ax.twinx()
    ax2.plot(xs, coh_y, color="grey", linestyle="--", linewidth=1.4, alpha=0.85,
             label="coherence")
    ax2.set_ylim(-3, 103)
    ax2.set_ylabel("coherence", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.axhline(COH_FLOOR, color="grey", linestyle=":", alpha=0.5, linewidth=0.8)

    # Shade layers that fail the coherence floor.
    in_fail = False
    for L in xs:
        bad = layers[L]["steer_coh"] < COH_FLOOR
        if bad and not in_fail:
            start = L - 0.5
            in_fail = True
        if not bad and in_fail:
            ax.axvspan(start, L - 0.5, color="grey", alpha=0.07)
            in_fail = False
    if in_fail:
        ax.axvspan(start, xs[-1] + 0.5, color="grey", alpha=0.07)

    # Reference lines and argmax markers.
    L_paper = PAPER_PICK
    ax.axvline(L_paper, color="black", linestyle=":", linewidth=1.3, alpha=0.75)
    ax.annotate("paper\nL*=16", xy=(L_paper, 100), xytext=(L_paper + 0.4, 95),
                fontsize=8, color="black", alpha=0.85)

    # Highlight the high-trait plateau (steer_trait >= max - 5, contiguous around argmax).
    L_lit = s["L_star"]
    peak_tr = layers[L_lit]["steer_trait"]
    plateau_xs = [L for L in xs if layers[L]["steer_trait"] >= peak_tr - 5]
    # restrict to a contiguous band around L_lit
    band_lo = L_lit
    while band_lo - 1 in plateau_xs:
        band_lo -= 1
    band_hi = L_lit
    while band_hi + 1 in plateau_xs:
        band_hi += 1
    ax.axvspan(band_lo - 0.4, band_hi + 0.4, color=colour, alpha=0.08, zorder=0)
    ax.text((band_lo + band_hi) / 2, 4,
            f"plateau (within 5 pts of peak): L={band_lo}–{band_hi}",
            ha="center", fontsize=7.5, color=colour, alpha=0.9)

    ax.plot([L_lit], [layers[L_lit]["steer_trait"]], marker="*", markersize=18,
            color="red", markeredgecolor="black", markeredgewidth=0.8, zorder=6,
            label=f"argmax(trait), no coh filter: L={L_lit}")

    L_coh = coh_aware_argmax(layers, COH_FLOOR)
    if L_coh is not None and L_coh != L_lit:
        ax.plot([L_coh], [layers[L_coh]["steer_trait"]], marker="*", markersize=18,
                color="gold", markeredgecolor="black", markeredgewidth=0.8, zorder=6,
                label=f"argmax(trait), coh≥50: L={L_coh}")

    # Gibberish callout for hallucinating L=32 (coh ≈ 0).
    if trait == "hallucinating":
        ax.annotate(
            "coh=0.01\n(gibberish)",
            xy=(32, layers[32]["steer_trait"]),
            xytext=(28, 50),
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8, alpha=0.7),
        )

    # Title.
    base_tr = s["baseline_trait"]
    ax.set_title(
        f"{trait}  (α={coef}, baseline trait={base_tr:.1f})",
        fontsize=10,
    )

    # Combined legend in upper-left of left axis.
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="lower left",
              frameon=True, framealpha=0.9)


def main() -> None:
    data = load()
    per_trait = data["per_trait"]
    traits = [t for t, _ in data["config"]["traits_and_coefs"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, trait in zip(axes, traits):
        plot_one(ax, trait, per_trait[trait])

    # Per-trait coh>=50 picks for the subtitle.
    picks_coh = {
        t: coh_aware_argmax(
            {int(L): v for L, v in per_trait[t]["layers"].items()}, COH_FLOOR
        )
        for t in traits
    }
    picks_lit = {t: per_trait[t]["L_star"] for t in traits}

    paper_lit = sum(1 for t in traits if picks_lit[t] == PAPER_PICK)
    paper_coh = sum(1 for t in traits if picks_coh[t] == PAPER_PICK)

    lit_str = ", ".join(f"{t[:4]}={picks_lit[t]}" for t in traits)
    coh_str = ", ".join(f"{t[:4]}={picks_coh[t]}" for t in traits)

    fig.suptitle(
        "Paper-faithful §B.4 layer-selection replication on Llama-3.1-8B-Instruct  "
        "—  job 488481, N=5/q, α per paper Tables 8/9/10\n"
        f"argmax(trait), no coh filter:  {lit_str}  →  {paper_lit}/3 land at L=16   "
        f"·   "
        f"argmax(trait), coh ≥ 50:  {coh_str}  →  {paper_coh}/3 land at L=16",
        fontsize=10, y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = FIG_DIR / "fig_layer_selection_paper_repl.pdf"
    png_path = FIG_DIR / "fig_layer_selection_paper_repl.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=160)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
