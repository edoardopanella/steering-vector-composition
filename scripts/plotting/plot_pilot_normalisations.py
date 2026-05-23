"""
Preliminary plots for the Phase 15 normalisation pilot (Riccardo).

Two figures, both 3-panel (one column per metric):
  (a) joint Δ_composition vs α          (trait gain)
  (b) joint coherence vs α               (cost)
  (c) Δ_composition × coherence / 100    (combined utility metric)

  fig_alpha_sweep_true_per_pair.{pdf,png}   — one line per pair, with
      Phase 12's normalize=False α=4 reference shown as a per-pair short
      dashed segment at x=4 (clearer than overlapping star markers).
  fig_alpha_sweep_true_aggregate.{pdf,png}  — single line = mean across the
      5 working pairs (power_seeking pair excluded; its trait_b is broken),
      with ±1 SEM band. Phase 12 reference shown as a single ★ with a
      visible legend entry.

All lines are normalize=True conditions only (the dose-response curve we're
calibrating). Auto-detects which α values have judged CSVs on disk — easy
to refresh as new CSVs land (e.g. after the α=4.5, 5.5 cluster job).

Run:
    python -m scripts.plotting.plot_pilot_normalisations
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_DIR = Path("results/composition_pilot_normalisations/Llama-3.1-8B-Instruct")
FIG_OUT_DIR = Path("results/composition_pilot_normalisations")

PAIRS = [
    ("formality", "humorous"),
    ("formality", "impolite"),
    ("apathetic", "confidence"),
    ("evil", "sycophantic"),
    ("apathetic", "impolite"),
    ("apathetic", "power_seeking"),
]
COS = {
    ("formality", "humorous"): -0.52,
    ("formality", "impolite"): -0.23,
    ("apathetic", "confidence"): 0.01,
    ("evil", "sycophantic"): 0.42,
    ("apathetic", "impolite"): 0.69,
    ("apathetic", "power_seeking"): -0.01,
}

# 6 distinct colours (tab10 minus colours that mute in print).
PAIR_COLOURS = {
    ("formality", "humorous"): "#1f77b4",
    ("formality", "impolite"): "#ff7f0e",
    ("apathetic", "confidence"): "#2ca02c",
    ("evil", "sycophantic"): "#d62728",
    ("apathetic", "impolite"): "#9467bd",
    ("apathetic", "power_seeking"): "#8c564b",
}


def _csv(name: str) -> Path:
    return CSV_DIR / name


def _read_if_scored(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "trait_a" not in df.columns or not df["trait_a"].notna().any():
        return None
    return df


def _means(df: pd.DataFrame) -> tuple[float, float, float, float]:
    return (
        df["trait_a"].mean(),
        df["trait_b"].mean(),
        df["composition"].mean(),
        df["coherence"].mean(),
    )


def _collect_true_sweep(a: str, b: str) -> dict[float, tuple[float, float, float, float]]:
    """Return {α: (trait_a, trait_b, composition, coherence)} for every
    normalize=True joint CSV that exists AND is scored. Auto-detects α."""
    a, b = sorted([a, b])
    out: dict[float, tuple[float, float, float, float]] = {}
    for p in CSV_DIR.glob(f"{a}__{b}_joint_true_alpha*.csv"):
        # parse α from filename
        try:
            alpha = float(p.stem.split("alpha")[-1])
        except ValueError:
            continue
        df = _read_if_scored(p)
        if df is None:
            continue
        out[alpha] = _means(df)
    return out


def _baseline(a: str, b: str) -> tuple[float, float, float, float] | None:
    df = _read_if_scored(_csv(f"{a}__{b}_baseline.csv"))
    return _means(df) if df is not None else None


def _ref_false_alpha4(a: str, b: str) -> tuple[float, float, float, float] | None:
    df = _read_if_scored(_csv(f"{a}__{b}_joint_false_alpha4.0.csv"))
    return _means(df) if df is not None else None


def _utility(d_comp: float, coh: float) -> float:
    """Combined operating-point utility = Δcomposition × coherence / 100.

    Both inputs are 0–100 scale, so the product is also 0–100 with a clean
    intuition: 100 = perfect gain at perfect coherence, 0 = no gain OR no
    coherence, negative = trait went the wrong way.
    """
    return d_comp * coh / 100


def _gather() -> tuple[list[dict], pd.DataFrame, dict[tuple[str, str], dict]]:
    """Walk the CSVs once and build:
        rows: flat list of dicts (one per (pair, α) cell) — for the per-pair plot
        agg:  DataFrame of mean ± sem aggregated over pairs at each α
        false_per_pair: {(a, b): {alpha: 4.0, d_comp, coh, util}} — Phase 12 refs
    """
    rows: list[dict] = []
    false_per_pair: dict[tuple[str, str], dict] = {}

    for a_raw, b_raw in PAIRS:
        a, b = sorted([a_raw, b_raw])
        cos = COS.get((a, b), COS.get((b, a), 0.0))

        base = _baseline(a, b)
        if base is None:
            continue
        _, _, comp0, _ = base

        sweep = _collect_true_sweep(a, b)
        for α, (tra, trb, comp, coh) in sweep.items():
            rows.append({
                "pair_a": a, "pair_b": b,
                "pair_label": f"{a}+{b}",
                "cos": cos,
                "alpha": α,
                "trait_a": tra, "trait_b": trb,
                "composition": comp, "coherence": coh,
                "delta_composition": comp - comp0,
                "utility": _utility(comp - comp0, coh),
            })

        ref = _ref_false_alpha4(a, b)
        if ref is not None:
            d_false = ref[2] - comp0
            false_per_pair[(a, b)] = {
                "alpha": 4.0,
                "delta_composition": d_false,
                "coherence": ref[3],
                "utility": _utility(d_false, ref[3]),
            }

    df = pd.DataFrame(rows)
    # Aggregate over the 5 working pairs (exclude apathetic+power_seeking).
    df_for_agg = df[df["pair_label"] != "apathetic+power_seeking"]
    agg = (
        df_for_agg
        .groupby("alpha")
        .agg(
            delta_composition_mean=("delta_composition", "mean"),
            delta_composition_sem=("delta_composition", lambda x: x.std() / np.sqrt(len(x))),
            coherence_mean=("coherence", "mean"),
            coherence_sem=("coherence", lambda x: x.std() / np.sqrt(len(x))),
            utility_mean=("utility", "mean"),
            utility_sem=("utility", lambda x: x.std() / np.sqrt(len(x))),
            n=("delta_composition", "count"),
        )
        .reset_index()
        .sort_values("alpha")
    )

    return rows, agg, false_per_pair


def _plot_per_pair(rows: list[dict], false_per_pair: dict, out_path: Path) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_d, ax_c, ax_u = axes

    for (a, b), pair_df in df.groupby(["pair_a", "pair_b"]):
        pair_df = pair_df.sort_values("alpha")
        cos = pair_df["cos"].iloc[0]
        colour = PAIR_COLOURS.get((a, b), PAIR_COLOURS.get((b, a), "k"))
        label = f"{a}+{b} (cos={cos:+.2f})"
        for ax, col in [(ax_d, "delta_composition"), (ax_c, "coherence"), (ax_u, "utility")]:
            ax.plot(pair_df["alpha"], pair_df[col], marker="o",
                    color=colour, label=label, linewidth=1.5, markersize=5)

        # Phase 12 reference: hollow ◇ marker in the pair colour at x=3.85
        # (slightly offset from the True_α=4 ● marker so the two are visually
        # side-by-side, not overlapping).
        ref = false_per_pair.get((a, b))
        if ref is not None:
            for ax, col in [(ax_d, "delta_composition"), (ax_c, "coherence"), (ax_u, "utility")]:
                ax.scatter(
                    3.85, ref[col],
                    marker="D", s=80, facecolors="white",
                    edgecolors=colour, linewidths=2.0, zorder=5,
                )

    # Manual reference handle for the diamond marker — a single legend entry.
    from matplotlib.lines import Line2D
    ref_handle = Line2D(
        [0], [0], marker="D", markersize=10, markerfacecolor="white",
        markeredgecolor="black", markeredgewidth=2.0, linestyle="",
        label="Phase 12 reference\n(normalize=False, α=4)\nshown at x=3.85 in pair colour",
    )

    ax_d.set_xlabel("α (normalize=True)")
    ax_d.set_ylabel("Δ composition (joint − baseline)")
    ax_d.set_title("(a) Joint trait gain")
    ax_d.axhline(0, color="gray", linewidth=0.5)
    ax_d.grid(alpha=0.3)

    ax_c.set_xlabel("α (normalize=True)")
    ax_c.set_ylabel("Joint coherence (0–100)")
    ax_c.set_title("(b) Coherence cost")
    ax_c.axhline(50, color="red", linewidth=0.7, linestyle="--", alpha=0.6)
    ax_c.text(4.05, 51, "Anthropic threshold (coh=50)", color="red",
              fontsize=8, alpha=0.7)
    ax_c.set_ylim(0, 100)
    ax_c.grid(alpha=0.3)

    ax_u.set_xlabel("α (normalize=True)")
    ax_u.set_ylabel("Utility = Δ composition × coherence / 100")
    ax_u.set_title("(c) Combined utility")
    ax_u.axhline(0, color="gray", linewidth=0.5)
    ax_u.grid(alpha=0.3)

    # Phase 12.5 operating-point marker — vertical dashed line at α=4.5.
    for ax in (ax_d, ax_c, ax_u):
        ax.axvline(4.5, color="#2ca02c", linewidth=1.5, linestyle="--",
                   alpha=0.7, zorder=0)
    op_handle = Line2D(
        [0], [0], color="#2ca02c", linewidth=1.5, linestyle="--",
        label="Phase 12.5 operating point\n(committed α = 4.5)",
    )

    # Legend on far right.
    handles, labels = ax_u.get_legend_handles_labels()
    handles.append(ref_handle)
    labels.append(ref_handle.get_label())
    handles.append(op_handle)
    labels.append(op_handle.get_label())
    ax_u.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                fontsize=8, frameon=False, title="pair (cos)")

    fig.suptitle(
        "Phase 15 pilot — per-pair dose-response under normalize=True\n"
        "green dashed line = Phase 12.5 committed operating point (α = 4.5)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    png = out_path.with_suffix(".png")
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")
    print(f"saved {png}")
    plt.close(fig)


def _plot_aggregate(agg: pd.DataFrame, false_per_pair: dict, out_path: Path) -> None:
    # Aggregate the False α=4 references across the 5 working pairs too.
    working = {k: v for k, v in false_per_pair.items() if k != ("apathetic", "power_seeking")}
    if working:
        d_vals = np.array([v["delta_composition"] for v in working.values()])
        c_vals = np.array([v["coherence"] for v in working.values()])
        u_vals = np.array([v["utility"] for v in working.values()])
        false_agg = {
            "delta_composition_mean": d_vals.mean(),
            "delta_composition_sem": d_vals.std() / np.sqrt(len(d_vals)),
            "coherence_mean": c_vals.mean(),
            "coherence_sem": c_vals.std() / np.sqrt(len(c_vals)),
            "utility_mean": u_vals.mean(),
            "utility_sem": u_vals.std() / np.sqrt(len(u_vals)),
        }
    else:
        false_agg = None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_d, ax_c, ax_u = axes

    panels = [
        (ax_d, "delta_composition", "Δ composition (joint − baseline)", "(a) Joint trait gain"),
        (ax_c, "coherence", "Joint coherence (0–100)", "(b) Coherence cost"),
        (ax_u, "utility", "Utility = Δ comp × coh / 100", "(c) Combined utility"),
    ]
    for ax, col, ylabel, title in panels:
        ax.errorbar(
            agg["alpha"], agg[f"{col}_mean"], yerr=agg[f"{col}_sem"],
            marker="o", color="#1f77b4", linewidth=2.0, markersize=7,
            capsize=4, label="normalize=True (mean ± SEM)",
        )
        ax.fill_between(
            agg["alpha"],
            agg[f"{col}_mean"] - agg[f"{col}_sem"],
            agg[f"{col}_mean"] + agg[f"{col}_sem"],
            color="#1f77b4", alpha=0.15,
        )

        if false_agg is not None:
            ax.errorbar(
                [4.0], [false_agg[f"{col}_mean"]], yerr=[false_agg[f"{col}_sem"]],
                marker="*", color="#d62728", markersize=18, capsize=4,
                linewidth=0, label="Phase 12 reference\n(normalize=False, α=4)",
                markeredgecolor="black", markeredgewidth=0.8,
            )

        ax.set_xlabel("α (normalize=True)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        # Phase 12.5 operating-point marker.
        ax.axvline(4.5, color="#2ca02c", linewidth=1.5, linestyle="--",
                   alpha=0.7, zorder=0, label="Phase 12.5 op. point (α=4.5)")
        if col == "coherence":
            ax.axhline(50, color="red", linewidth=0.7, linestyle="--", alpha=0.6)
            ax.text(4.05, 51, "Anthropic threshold (coh=50)",
                    color="red", fontsize=8, alpha=0.7)
            ax.set_ylim(0, 100)
        elif col == "delta_composition":
            ax.axhline(0, color="gray", linewidth=0.5)
        elif col == "utility":
            ax.axhline(0, color="gray", linewidth=0.5)

        ax.legend(loc="best", fontsize=9, frameon=True)

    fig.suptitle(
        "Phase 15 pilot — aggregate dose-response under normalize=True\n"
        f"mean across {int(agg['n'].iloc[0])} pairs (apathetic+power_seeking excluded — broken trait); "
        "green dashed line = Phase 12.5 committed operating point (α = 4.5)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    png = out_path.with_suffix(".png")
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")
    print(f"saved {png}")
    plt.close(fig)


def main() -> None:
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, agg, false_per_pair = _gather()
    if not rows:
        print("No data found. Run the cluster generate + laptop judge first.")
        return

    # Sidecar data CSVs
    pd.DataFrame(rows).sort_values(["pair_label", "alpha"]).to_csv(
        FIG_OUT_DIR / "fig_alpha_sweep_true_data.csv", index=False
    )
    agg.to_csv(FIG_OUT_DIR / "fig_alpha_sweep_true_aggregate_data.csv", index=False)

    _plot_per_pair(rows, false_per_pair, FIG_OUT_DIR / "fig_alpha_sweep_true_per_pair.pdf")
    _plot_aggregate(agg, false_per_pair, FIG_OUT_DIR / "fig_alpha_sweep_true_aggregate.pdf")

    # Remove the old combined figure name from prior runs so we don't keep
    # showing a stale version.
    for stale in ("fig_alpha_sweep_true.pdf", "fig_alpha_sweep_true.png"):
        p = FIG_OUT_DIR / stale
        if p.exists():
            p.unlink()
            print(f"removed stale {p}")


if __name__ == "__main__":
    main()
