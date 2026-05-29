"""
Plots for the exact MRQAP / Mantel geometry -> composition analysis.

Imports the validated machinery from rq1_mrqap_significance.py so the figures use
exactly the numbers in the result tables (no re-derivation drift). Produces, per
controlled scheme (normalized_sum, per_axis):

  fig_mrqap_matrices         8x8 COS / Smat / Mmat heatmaps (gated dyads marked)
  fig_mrqap_direction_scatter  Smat vs signed cos and vs |cos| (sign-matters)
  fig_mrqap_magnitude_scatter  Mmat vs cos (the magnitude "null" that isn't)
  fig_mrqap_perm_null        exact 40,320-perm null for the headline Smat~COS|[STR]
  fig_mrqap_loo_leverage     leave-one-trait-out beta per dropped trait

Run:  ./venv/bin/python analysis/rq1_mrqap_plots.py
"""
from pathlib import Path
from itertools import permutations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from rq1_mrqap_significance import (
    load_frames, build_matrices, utri, partial_slope, _valid_mask,
    bivariate_r, mrqap_exact, gate_report, SCHEME_LABEL, CONTROLLED, COH_GATE, REPO,
)

FIGDIR = REPO / "analysis/figures/mrqap"
COLOR = {"normTrue": "#2166ac", "per_axis": "#d6604d"}
ABBR = {"apathetic": "apat", "confidence": "conf", "evil": "evil", "formality": "form",
        "hallucinating": "hall", "humorous": "humo", "impolite": "impo", "sycophantic": "syco"}


def savefig(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {FIGDIR.relative_to(REPO)}/{name}.png (+ .pdf)")


def null_dist(OUT, PRED, CTRLS, base_mask):
    """Full exact permutation null of the standardized partial slope."""
    n = OUT.shape[0]
    iu = np.triu_indices(n, k=1)
    y, predf = OUT[iu], PRED[iu]
    ctrlf = [C[iu] for C in CTRLS]
    obs_valid = _valid_mask(base_mask, predf, y, ctrlf)
    beta_obs = partial_slope(y, predf, ctrlf, obs_valid)
    nulls = []
    for p in permutations(range(n)):
        px = PRED[np.ix_(p, p)][iu]
        valid = _valid_mask(base_mask, px, y, ctrlf)
        b = partial_slope(y, px, ctrlf, valid)
        if np.isfinite(b):
            nulls.append(b)
    nulls = np.asarray(nulls)
    p_two = float(np.mean(np.abs(nulls) >= abs(beta_obs) - 1e-12))
    return beta_obs, nulls, p_two


def scatter_panel(ax, x, y, color, line=True):
    sc = ax.scatter(x, y, c=color, s=46, edgecolor="white", linewidth=0.6, zorder=3)
    if line and len(x) >= 2 and np.std(x) > 0:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m * xs + b, color="0.25", lw=1.6, zorder=2)
    return sc


# ---------------------------------------------------------------------------
def fig_matrices(mats, traits):
    labels = [ABBR[t] for t in traits]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2))
    specs = [("COS", "RdBu_r", True), ("Smat (direction)", "RdBu_r", True),
             ("Mmat (magnitude)", "viridis", False)]
    for ri, scheme in enumerate(CONTROLLED):
        COS, Smat, Mmat, STR, _ = mats[scheme]
        mtx = {"COS": COS, "Smat (direction)": Smat, "Mmat (magnitude)": Mmat}
        for ci, (title, cmap, diverging) in enumerate(specs):
            ax = axes[ri, ci]
            M = np.ma.masked_invalid(mtx[title])
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("0.85")
            if diverging:
                vmax = np.nanmax(np.abs(mtx[title]))
                im = ax.imshow(M, cmap=cm, vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(M, cmap=cm)
            # mark gated (NaN, off-diagonal) cells with an x
            for i in range(len(traits)):
                for j in range(len(traits)):
                    if i != j and not np.isfinite(mtx[title][i, j]):
                        ax.text(j, i, "×", ha="center", va="center", color="0.45", fontsize=9)
            ax.set_xticks(range(len(traits)))
            ax.set_yticks(range(len(traits)))
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(f"{SCHEME_LABEL[scheme]}\n{title}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("8×8 trait dyad matrices per scheme  (× = coherence-gated dyad, dropped from the test)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, "fig_mrqap_matrices")


# ---------------------------------------------------------------------------
def fig_direction_scatter(mats):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey="row")
    for ri, scheme in enumerate(CONTROLLED):
        COS, Smat, Mmat, STR, base_mask = mats[scheme]
        iu = np.triu_indices(COS.shape[0], 1)
        valid = np.isfinite(utri(COS)) & np.isfinite(utri(Smat))
        cos_v, acos_v, s_v = utri(COS)[valid], np.abs(utri(COS))[valid], utri(Smat)[valid]
        # partial (control STR) exact p for signed; raw Mantel p for both
        b_par, p_par, _, _ = mrqap_exact(Smat, COS, [STR], base_mask)
        _, p_raw_signed, _, _ = mrqap_exact(Smat, COS, [], base_mask)
        _, p_raw_abs, _, _ = mrqap_exact(Smat, np.abs(COS), [], base_mask)
        r_signed = bivariate_r(Smat, COS, base_mask)
        r_abs = bivariate_r(Smat, np.abs(COS), base_mask)

        for ci, (xv, xlab, r, p_raw, extra) in enumerate([
            (cos_v, "signed cosine", r_signed, p_raw_signed,
             f" · partial β|STR = {b_par:+.2f} (p = {p_par:.3f})"),
            (acos_v, "|cosine|", r_abs, p_raw_abs, "")]):
            ax = axes[ri, ci]
            scatter_panel(ax, xv, s_v, COLOR[scheme])
            ax.axhline(0, color="0.5", lw=1, ls="--")
            ax.set_xlabel(xlab)
            if ci == 0:
                ax.set_ylabel(f"{SCHEME_LABEL[scheme]}\n\nSmat  (joint − individual)")
            col = "vs signed cosine — sign matters" if ci == 0 else "vs |cosine| — ≈ null"
            stat = f"Mantel r = {r:+.2f} (p = {p_raw:.3f}){extra}"
            ax.set_title((col + "\n" if ri == 0 else "") + stat, fontsize=9)
    fig.suptitle("Directional outcome vs geometry — sign of cosine carries the signal "
                 "(Smat > 0 reinforce, < 0 suppress)", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, "fig_mrqap_direction_scatter")


# ---------------------------------------------------------------------------
def fig_magnitude_scatter(mats):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ci, scheme in enumerate(CONTROLLED):
        COS, Smat, Mmat, STR, base_mask = mats[scheme]
        valid = np.isfinite(utri(COS)) & np.isfinite(utri(Mmat))
        cos_v, m_v = utri(COS)[valid], utri(Mmat)[valid]
        b_par, p_par, _, _ = mrqap_exact(Mmat, COS, [STR], base_mask)
        _, p_raw, _, _ = mrqap_exact(Mmat, COS, [], base_mask)
        r = bivariate_r(Mmat, COS, base_mask)
        ax = axes[ci]
        scatter_panel(ax, cos_v, m_v, COLOR[scheme])
        ax.set_xlabel("signed cosine")
        if ci == 0:
            ax.set_ylabel("Mmat  (joint expression magnitude)")
        ax.set_title(f"{SCHEME_LABEL[scheme]}\n"
                     f"Mantel r = {r:+.2f} (p = {p_raw:.3f}) · "
                     f"partial β|STR = {b_par:+.2f} (p = {p_par:.3f})", fontsize=9)
    fig.suptitle("Joint magnitude is NOT a clean null — it tracks cosine alignment too",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig_mrqap_magnitude_scatter")


# ---------------------------------------------------------------------------
def fig_perm_null(mats):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ci, scheme in enumerate(CONTROLLED):
        COS, Smat, Mmat, STR, base_mask = mats[scheme]
        beta_obs, nulls, p_two = null_dist(Smat, COS, [STR], base_mask)
        ax = axes[ci]
        ax.hist(nulls, bins=60, color="0.8", edgecolor="0.6", lw=0.3)
        tail = nulls[np.abs(nulls) >= abs(beta_obs) - 1e-12]
        ax.hist(tail, bins=60, color=COLOR[scheme], alpha=0.75,
                label=f"|β| ≥ |β_obs|  (p = {p_two:.3f})")
        ax.axvline(beta_obs, color="k", lw=2, label=f"observed β = {beta_obs:+.2f}")
        ax.axvline(-beta_obs, color="k", lw=1, ls=":")
        ax.set_xlabel("standardized partial slope  (β_cos | STR)")
        if ci == 0:
            ax.set_ylabel("permutations")
        ax.set_title(f"{SCHEME_LABEL[scheme]}  ({len(nulls):,} perms)", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Exact MRQAP null for the headline directional test  (Smat ~ COS | STR)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig_mrqap_perm_null")


# ---------------------------------------------------------------------------
def fig_loo_leverage(gated, traits):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True)
    for ci, scheme in enumerate(CONTROLLED):
        df_s = gated[gated["scheme"] == scheme].reset_index(drop=True)
        COS, Smat, Mmat, STR = build_matrices(df_s, traits)
        base_mask = np.isfinite(utri(COS))
        b_full, p_full, _, _ = mrqap_exact(Smat, COS, [STR], base_mask)
        rows = [("(full, 8 traits)", b_full, p_full)]
        for drop in traits:
            keep = [t for t in traits if t != drop]
            sub = df_s[(df_s["trait_a"] != drop) & (df_s["trait_b"] != drop)]
            cS = build_matrices(sub, keep)
            bm = np.isfinite(utri(cS[0]))
            try:
                b, p, _, _ = mrqap_exact(cS[1], cS[0], [cS[3]], bm)
            except AssertionError:
                b, p = np.nan, np.nan
            rows.append((f"− {drop}", b, p))
        ax = axes[ci]
        ys = np.arange(len(rows))[::-1]
        for y, (lab, b, p) in zip(ys, rows):
            if not np.isfinite(b):
                continue
            sig = p < 0.05
            ax.scatter(b, y, s=120 if lab.startswith("(full") else 70,
                       color=COLOR[scheme] if sig else "white",
                       edgecolor=COLOR[scheme], linewidth=1.6, zorder=3,
                       marker="D" if lab.startswith("(full") else "o")
            ax.text(b, y + 0.18, f"p={p:.3f}", ha="center", va="bottom", fontsize=7.5, color="0.3")
        ax.axvline(b_full, color=COLOR[scheme], lw=1, ls="--", alpha=0.6)
        ax.axvline(0, color="0.6", lw=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.set_xlabel("β_std_partial  (Smat ~ COS | STR)")
        ax.set_title(SCHEME_LABEL[scheme], fontsize=11)
    leg = [Patch(fc=COLOR["normTrue"], ec=COLOR["normTrue"], label="p < 0.05 (filled)"),
           Patch(fc="white", ec="0.4", label="p ≥ 0.05 (open)")]
    axes[1].legend(handles=leg, fontsize=8, loc="lower right")
    fig.suptitle("Leave-one-trait-out leverage on the directional effect", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig_mrqap_loo_leverage")


def main():
    gated, raw, traits = load_frames()
    mats = {}
    for scheme in CONTROLLED:
        df_s = gated[gated["scheme"] == scheme].reset_index(drop=True)
        COS, Smat, Mmat, STR = build_matrices(df_s, traits)
        base_mask = np.isfinite(utri(COS))
        mats[scheme] = (COS, Smat, Mmat, STR, base_mask)

    fig_matrices(mats, traits)
    fig_direction_scatter(mats)
    fig_magnitude_scatter(mats)
    fig_perm_null(mats)
    fig_loo_leverage(gated, traits)
    print(f"\nAll figures written to {FIGDIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
