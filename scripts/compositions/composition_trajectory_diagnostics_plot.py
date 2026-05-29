"""
Trajectory-side diagnostic plots for the composition-scoring run.

Loads results/composition/v1_phase12_normFalse_a4/trajectories/aggregate.parquet (long-form table of
π_i^(α_i, α_j)(L) values across 36 pairs × 3 settings × 100 prompts × 16 layers
× 2 behaviour axes) plus the τ JSON and the per-pair regime metadata. Produces
one multi-panel PDF for sanity-checking the RQ2 Phase 2 dataset BEFORE writing
any inferential trajectory analysis (Phase 3 Δ / L_div / dual-projection).

Quantities derived per pair × per behaviour axis:
  Δ_i(i,j) = mean over L of |π_i^(1,1)(L) − π_i^(1,0)(L)|     (Eq 2, RQ2 PDF)
  L_div(i,j) = min L : |π_i^(1,1)(L) − π_i^(1,0)(L)| > τ      (Eq 3, RQ2 PDF)

Panels:
  A. Δ distribution per axis     (histogram, axis_a vs axis_b stacked)
  B. Δ by regime, axis a         (boxplot stratified by regime; expected Phase 3 result)
  C. Δ by regime, axis b         (same, mirror)
  D. L_div by regime, axis a     (boxplot; right-censored at L_final when never crossed)
  E. L_div by regime, axis b
  F. Per-pair |Δπ(L)| heat-map   (rows = pairs sorted by regime, cols = layers; sanity for τ scale)

Run:
    python -m scripts.compositions.composition_trajectory_diagnostics_plot

Output:
    results/figures/composition_trajectory_diagnostics.pdf
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AGG_PARQUET = Path("results/composition/v1_phase12_normFalse_a4/trajectories/aggregate.parquet")
TAU_PATH = Path("results/composition/v1_phase12_normFalse_a4/trajectories/tau.json")
SUMMARY_PATH = Path("results/composition/v1_phase12_normFalse_a4/scoring/summary.json")
FIG_OUT = Path("results/figures/composition_trajectory_diagnostics.pdf")

ALPHA = 4.0
HIDDEN_LAYER = 17

REGIME_COLORS = {
    "additive": "#0072B2",
    "dominant": "#E69F00",
    "suppressive": "#CC79A7",
    "emergent": "#009E73",
    "mixed": "#999999",
}
REGIME_ORDER = ["additive", "dominant", "suppressive", "emergent", "mixed"]


def _compute_per_pair_deltas(agg: pd.DataFrame, tau: float) -> pd.DataFrame:
    """For every (pair_id, behaviour_index) compute Δ and L_div from the
    aggregate parquet. Joint (α_i=ALPHA, α_j=ALPHA) compared to the matching
    single-axis condition: axis 0 -> single_a (ALPHA, 0); axis 1 -> single_b (0, ALPHA).

    Returns long-form: pair_id, behaviour_index, trait_i, trait_j, regime, cosine,
    delta, l_div (NaN if never crossed τ — interpret as right-censored at L_final).
    """
    L_final = int(agg["layer"].max())
    # Means per (pair, behaviour, alpha_i, alpha_j, layer): average over prompts.
    g = (
        agg.groupby(
            ["pair_id", "behaviour_index", "alpha_i", "alpha_j", "layer"],
            sort=False,
        )["projection_value"].mean().reset_index()
    )
    # Pivot per (pair, behaviour, layer): columns by setting.
    out_rows: list[dict] = []
    meta = agg.drop_duplicates("pair_id")[
        ["pair_id", "trait_i", "trait_j", "regime", "cosine"]
    ].set_index("pair_id")
    for (pair_id, beh), sub in g.groupby(["pair_id", "behaviour_index"], sort=False):
        joint = sub[(sub.alpha_i == ALPHA) & (sub.alpha_j == ALPHA)].set_index("layer")["projection_value"]
        if beh == 0:
            single = sub[(sub.alpha_i == ALPHA) & (sub.alpha_j == 0.0)].set_index("layer")["projection_value"]
        else:
            single = sub[(sub.alpha_i == 0.0) & (sub.alpha_j == ALPHA)].set_index("layer")["projection_value"]
        layers = sorted(set(joint.index) & set(single.index))
        diff = np.array([abs(joint[L] - single[L]) for L in layers])
        delta = float(diff.mean()) if len(diff) else float("nan")
        cross = [L for L, d in zip(layers, diff) if d > tau]
        l_div = float(min(cross)) if cross else float("nan")
        m = meta.loc[pair_id]
        out_rows.append({
            "pair_id": int(pair_id),
            "behaviour_index": int(beh),
            "trait_i": m["trait_i"],
            "trait_j": m["trait_j"],
            "regime": m["regime"],
            "cosine": float(m["cosine"]),
            "delta": delta,
            "l_div": l_div,
            "max_diff": float(diff.max()) if len(diff) else float("nan"),
        })
    return pd.DataFrame(out_rows)


def _panel_delta_hist(ax, dl: pd.DataFrame, tau: float) -> None:
    bins = np.linspace(0, dl["delta"].max() * 1.05 + 1e-6, 25)
    ax.hist(
        [dl[dl.behaviour_index == 0]["delta"], dl[dl.behaviour_index == 1]["delta"]],
        bins=bins, stacked=False,
        label=["axis a (trait_i)", "axis b (trait_j)"],
        color=["#0072B2", "#E69F00"], edgecolor="black", linewidth=0.4,
    )
    ax.axvline(tau, ls="--", color="red", lw=1.0, label=f"τ = {tau:.3f}")
    ax.set_xlabel("Δ  (mean over L of |π_joint − π_single|)")
    ax.set_ylabel("# pairs")
    ax.set_title("A. Δ distribution per axis", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False)


def _panel_delta_by_regime(ax, dl: pd.DataFrame, axis: int, label: str) -> None:
    data, ticks, colors = [], [], []
    for r in REGIME_ORDER:
        sub = dl[(dl.regime == r) & (dl.behaviour_index == axis)]["delta"].dropna()
        if len(sub):
            data.append(sub.values)
            ticks.append(f"{r}\n(n={len(sub)})")
            colors.append(REGIME_COLORS[r])
    bp = ax.boxplot(data, tick_labels=ticks, patch_artist=True, widths=0.55)
    for box, c in zip(bp["boxes"], colors):
        box.set(facecolor=c, alpha=0.7, edgecolor="black", linewidth=0.5)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.2)
    ax.set_ylabel(f"Δ axis {'a' if axis == 0 else 'b'}")
    ax.set_title(f"{label}. Δ by regime, axis {'a' if axis == 0 else 'b'}",
                 fontsize=11, loc="left")
    # Per-pair points overlay.
    for i, vals in enumerate(data, 1):
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.08, 0.08, len(vals)),
                   vals, s=14, color="black", alpha=0.5, zorder=3)


def _panel_ldiv_by_regime(ax, dl: pd.DataFrame, axis: int, L_final: int, label: str) -> None:
    data, ticks, colors = [], [], []
    for r in REGIME_ORDER:
        sub = dl[(dl.regime == r) & (dl.behaviour_index == axis)].copy()
        if not len(sub):
            continue
        # right-censor: never crossed -> L_final + 1 marker
        l_div_plot = sub["l_div"].fillna(L_final + 1).values
        data.append(l_div_plot)
        ticks.append(f"{r}\n(n={len(sub)})")
        colors.append(REGIME_COLORS[r])
    bp = ax.boxplot(data, tick_labels=ticks, patch_artist=True, widths=0.55)
    for box, c in zip(bp["boxes"], colors):
        box.set(facecolor=c, alpha=0.7, edgecolor="black", linewidth=0.5)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.2)
    ax.axhline(L_final + 0.5, color="red", ls=":", lw=0.8, label=f"never crossed τ (= {L_final + 1})")
    ax.set_ylabel(f"L_div axis {'a' if axis == 0 else 'b'}")
    ax.set_title(f"{label}. L_div by regime, axis {'a' if axis == 0 else 'b'}",
                 fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for i, vals in enumerate(data, 1):
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.08, 0.08, len(vals)),
                   vals, s=14, color="black", alpha=0.5, zorder=3)


def _panel_heatmap(ax, agg: pd.DataFrame, dl: pd.DataFrame, tau: float) -> None:
    """|π_joint − π_single| heatmap, rows = pairs sorted by regime then |cos|,
    cols = layers, averaged over axes and prompts."""
    layers = sorted(agg["layer"].unique())
    # mean per (pair, axis, layer) at joint and matching single
    g = (
        agg.groupby(["pair_id", "behaviour_index", "alpha_i", "alpha_j", "layer"], sort=False)
        ["projection_value"].mean().reset_index()
    )
    rows, row_labels, row_colors = [], [], []
    meta = agg.drop_duplicates("pair_id")[["pair_id", "trait_i", "trait_j", "regime", "cosine"]]
    meta["abscos"] = meta["cosine"].abs()
    meta["regime_rank"] = meta["regime"].map({r: i for i, r in enumerate(REGIME_ORDER)})
    meta = meta.sort_values(["regime_rank", "abscos"]).reset_index(drop=True)
    for _, m in meta.iterrows():
        pid = int(m["pair_id"])
        sub = g[g.pair_id == pid]
        # average per layer of mean(axis 0, axis 1) |joint − single|
        diffs = []
        for L in layers:
            d = []
            for beh in (0, 1):
                joint = sub[(sub.behaviour_index == beh) & (sub.alpha_i == ALPHA) & (sub.alpha_j == ALPHA) & (sub.layer == L)]["projection_value"]
                if beh == 0:
                    single = sub[(sub.behaviour_index == beh) & (sub.alpha_i == ALPHA) & (sub.alpha_j == 0.0) & (sub.layer == L)]["projection_value"]
                else:
                    single = sub[(sub.behaviour_index == beh) & (sub.alpha_i == 0.0) & (sub.alpha_j == ALPHA) & (sub.layer == L)]["projection_value"]
                if len(joint) and len(single):
                    d.append(abs(joint.values[0] - single.values[0]))
            diffs.append(np.mean(d) if d else 0.0)
        rows.append(diffs)
        row_labels.append(f"{m['trait_i'][:4]}+{m['trait_j'][:4]} {m['cosine']:+.2f} [{m['regime'][:3]}]")
        row_colors.append(REGIME_COLORS[m["regime"]])
    mat = np.array(rows)
    im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower",
                   extent=[layers[0] - 0.5, layers[-1] + 0.5, -0.5, len(rows) - 0.5])
    ax.set_xlabel("layer")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=6)
    for tl, c in zip(ax.get_yticklabels(), row_colors):
        tl.set_color(c)
    ax.set_title(f"F. mean over axes of |π_joint − π_single|(L), τ = {tau:.3f}",
                 fontsize=11, loc="left")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="|Δπ|")


def main() -> None:
    if not AGG_PARQUET.exists():
        raise SystemExit(f"Aggregate parquet missing: {AGG_PARQUET}. Run composition_aggregate_local.py first.")
    if not TAU_PATH.exists():
        raise SystemExit(f"τ JSON missing: {TAU_PATH}.")
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"Summary JSON missing: {SUMMARY_PATH}.")

    agg = pd.read_parquet(AGG_PARQUET)
    tau_info = json.loads(TAU_PATH.read_text())
    tau = float(tau_info["tau"])
    L_final = int(agg["layer"].max())

    print(f"agg rows={len(agg):,}  pairs={agg['pair_id'].nunique()}  layers=[{agg['layer'].min()}, {L_final}]")
    print(f"τ = {tau:.4f}")

    dl = _compute_per_pair_deltas(agg, tau)
    print(f"\nΔ stats per axis:")
    for axis in (0, 1):
        s = dl[dl.behaviour_index == axis]["delta"]
        print(f"  axis {axis}: n={len(s)}  mean={s.mean():.3f}  median={s.median():.3f}  "
              f"min={s.min():.3f}  max={s.max():.3f}")
    crossed = dl["l_div"].notna().sum()
    total = len(dl)
    print(f"\nL_div crossings: {crossed}/{total} pair-axes ever exceed τ")
    print(f"  never-crossed (right-censored): {total - crossed}")

    # Per regime: count crossings + median L_div
    print("\nper-regime: # crossed (out of n) | median L_div (NaN if no crossings)")
    for r in REGIME_ORDER:
        for axis in (0, 1):
            sub = dl[(dl.regime == r) & (dl.behaviour_index == axis)]
            n = len(sub)
            c = sub["l_div"].notna().sum()
            med = sub["l_div"].median()
            print(f"  {r:<11} axis{axis}: {c}/{n}  median L_div = {'NaN' if pd.isna(med) else f'{med:.1f}'}")

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2.0], hspace=0.4, wspace=0.25)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0])
    d = fig.add_subplot(gs[1, 1])
    e = fig.add_subplot(gs[2, :])
    _panel_delta_hist(a, dl, tau)
    _panel_delta_by_regime(b, dl, 0, "B")
    _panel_delta_by_regime(c, dl, 1, "C")
    _panel_ldiv_by_regime(d, dl, 0, L_final, "D")
    # Re-layout: panel E becomes L_div axis b (replace e with split bottom)
    fig.delaxes(e)
    gs2 = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2.0], hspace=0.4, wspace=0.25)
    e1 = fig.add_subplot(gs2[2, 0])
    e2 = fig.add_subplot(gs2[2, 1])
    _panel_ldiv_by_regime(e1, dl, 1, L_final, "E")
    _panel_heatmap(e2, agg, dl, tau)
    fig.suptitle(
        f"Trajectory diagnostics — L*=17, α={ALPHA}, τ={tau:.3f}  (n={agg['pair_id'].nunique()} pairs × 2 axes)",
        fontsize=13, y=0.995,
    )
    fig.savefig(FIG_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {FIG_OUT}  ({FIG_OUT.stat().st_size / 1024:.0f} KB)")

    # Also save the derived Δ/L_div table for downstream scripts.
    out_csv = Path("results/composition/v1_phase12_normFalse_a4/trajectories/composition_delta_ldiv.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dl.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(dl)} rows)")


if __name__ == "__main__":
    main()
