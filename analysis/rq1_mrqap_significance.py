"""
Exact trait-level MRQAP / Mantel significance for the geometry -> composition analysis.

Run PER controlled scheme (normalized_sum == normTrue, per_axis). This replaces the
invalid within-scheme pair-label shuffle over n=76 pooled pair x scheme rows
(which ignores dyadic dependence and double-counts each pair's cosine across schemes)
with an EXACT 8! = 40,320 node-permutation test on 8x8 trait matrices.

Outcomes / predictors / controls follow the brief:
    Smat[i,j] = 0.5 * ((d_i^joint - d_i^ind) + (d_j^joint - d_j^ind))   (directional, signed)
    Mmat[i,j] = 0.5 * (|d_i^joint| + |d_j^joint|)                       (joint magnitude, ~null)
    STR [i,j] = 0.5 * (|d_i^ind|   + |d_j^ind|)                         (individual-strength control)
    COS [i,j] = signed cosine of the two unit steering directions
where d_i^joint = E_i^(1,1) - E_i^(0,0)  and  d_i^ind = E_i^(1,0) - E_i^(0,0).

Data: analysis/rq1_consolidated/consolidated_coh30.csv  (the frame used by
analysis/notebooks/rq1_v2v3_decomposition.ipynb; coherence gate coh_steered >= 30
is already baked in). Deltas map directly to CSV columns:
    d_a^joint = delta_a_joint, d_a^ind = delta_a_single (and b analogues);
    Mmat == mean_joint_abs, STR == mean_single_abs, COS == cos.

NaN / gate handling under node relabeling
-----------------------------------------
The gate removes whole dyads (per_axis loses 6 of 28; normTrue loses 0). Gated dyads are
NaN in EVERY matrix for that scheme, so all matrices share one valid-edge mask. Under a
node permutation those NaN cells move, so for per_axis a naive fixed-mask extraction would
pull NaNs into the statistic. We therefore use AVAILABLE-CASE node permutation: for each
relabeling the statistic is computed on dyads observed in both predictor and outcome
(controls refit on that set). This is the standard QAP treatment of structurally missing
dyads; it reduces EXACTLY to the brief's fixed-mask code when no dyad is gated (normTrue),
which the asserts verify. Permutations that yield an undefined slope are dropped from the
reference distribution (not silently counted as non-exceeding), so the p-value is not
deflated. The observed (identity) statistic always uses the full valid edge set.
"""
from pathlib import Path
from itertools import permutations

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV_GATED = REPO / "analysis/rq1_consolidated/consolidated_coh30.csv"
CSV_RAW = REPO / "analysis/rq1_consolidated/consolidated_raw.csv"

COH_GATE = 30.0                       # coh_steered >= 30 (already applied to consolidated_coh30.csv)
SCHEME_LABEL = {"normTrue": "normalized_sum", "per_axis": "per_axis"}
CONTROLLED = ["normTrue", "per_axis"]  # the two dose-controlled schemes; 'normFalse' (naive) excluded


# ---------------------------------------------------------------------------
# data loading + matrix construction
# ---------------------------------------------------------------------------
def load_frames():
    gated = pd.read_csv(CSV_GATED)
    raw = pd.read_csv(CSV_RAW)
    gated = gated[gated["scheme"].isin(CONTROLLED)].reset_index(drop=True)
    raw = raw[raw["scheme"].isin(CONTROLLED)].reset_index(drop=True)
    traits = sorted(set(gated["trait_a"]).union(gated["trait_b"]))
    return gated, raw, traits


def build_matrices(df_scheme, traits):
    """Return symmetric 8x8 COS, Smat, Mmat, STR (NaN off the valid-edge mask) for one scheme."""
    n = len(traits)
    idx = {t: i for i, t in enumerate(traits)}
    COS = np.full((n, n), np.nan)
    Smat = np.full((n, n), np.nan)
    Mmat = np.full((n, n), np.nan)
    STR = np.full((n, n), np.nan)
    for _, r in df_scheme.iterrows():
        i, j = idx[r["trait_a"]], idx[r["trait_b"]]
        dasj, dasi = r["delta_a_joint"], r["delta_a_single"]
        dbsj, dbsi = r["delta_b_joint"], r["delta_b_single"]
        cos = r["cos"]
        s = 0.5 * ((dasj - dasi) + (dbsj - dbsi))
        m = 0.5 * (abs(dasj) + abs(dbsj))
        t = 0.5 * (abs(dasi) + abs(dbsi))
        for a, b in ((i, j), (j, i)):
            COS[a, b], Smat[a, b], Mmat[a, b], STR[a, b] = cos, s, m, t
    return COS, Smat, Mmat, STR


def utri(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


# ---------------------------------------------------------------------------
# faithful brief primitives (used only for a fidelity assert)
# ---------------------------------------------------------------------------
def resid_on(y, Zcols):
    X = np.column_stack([np.ones_like(y)] + list(Zcols))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def std_slope(rx, ry):
    return np.polyfit(rx / rx.std(), ry / ry.std(), 1)[0]


# ---------------------------------------------------------------------------
# available-case standardized partial slope (== partial correlation)
# ---------------------------------------------------------------------------
def partial_slope(y, x, ctrl_cols, valid):
    """Standardized slope of OUT~PRED after partialling controls, on `valid` edges only."""
    m = int(valid.sum())
    if m < len(ctrl_cols) + 3:
        return np.nan
    yv, xv = y[valid], x[valid]
    if ctrl_cols:
        Z = np.column_stack([np.ones(m)] + [c[valid] for c in ctrl_cols])
        Ginv = np.linalg.pinv(Z.T @ Z)
        ry = yv - Z @ (Ginv @ (Z.T @ yv))
        rx = xv - Z @ (Ginv @ (Z.T @ xv))
    else:
        ry = yv - yv.mean()
        rx = xv - xv.mean()
    nx, ny = np.sqrt(rx @ rx), np.sqrt(ry @ ry)
    if nx == 0 or ny == 0:
        return np.nan
    return float((rx @ ry) / (nx * ny))   # == np.polyfit(rx/rx.std(), ry/ry.std(), 1)[0]


def _valid_mask(base_mask, px, y, ctrl_cols):
    valid = base_mask & np.isfinite(px) & np.isfinite(y)
    for c in ctrl_cols:
        valid = valid & np.isfinite(c)
    return valid


def mrqap_exact(OUT, PRED, CTRLS, base_mask):
    """Exact node-permutation MRQAP. Returns (beta_obs, two_sided_p, n_edges_used, n_perms_used)."""
    n = OUT.shape[0]
    iu = np.triu_indices(n, k=1)
    y = OUT[iu]
    predf = PRED[iu]
    ctrlf = [C[iu] for C in CTRLS]
    obs_valid = _valid_mask(base_mask, predf, y, ctrlf)
    beta_obs = partial_slope(y, predf, ctrlf, obs_valid)
    n_edges = int(obs_valid.sum())

    fixed = bool(base_mask.all() and np.isfinite(predf).all())  # normTrue: no gated dyads
    count = n_used = 0
    first_b = None
    if fixed:
        m = int(obs_valid.sum())
        if ctrlf:
            Z = np.column_stack([np.ones(m)] + [c[obs_valid] for c in ctrlf])
            Ginv = np.linalg.pinv(Z.T @ Z)

            def resid(vfull):
                xv = vfull[obs_valid]
                return xv - Z @ (Ginv @ (Z.T @ xv))
        else:
            def resid(vfull):
                xv = vfull[obs_valid]
                return xv - xv.mean()
        ry = resid(y)
        ny = np.sqrt(ry @ ry)
        for k, p in enumerate(permutations(range(n))):
            px = PRED[np.ix_(p, p)][iu]
            rx = resid(px)
            nx = np.sqrt(rx @ rx)
            b = (rx @ ry) / (nx * ny) if (nx > 0 and ny > 0) else np.nan
            if k == 0:
                first_b = b
            if np.isfinite(b):
                n_used += 1
                if abs(b) >= abs(beta_obs) - 1e-12:
                    count += 1
    else:
        for k, p in enumerate(permutations(range(n))):
            px = PRED[np.ix_(p, p)][iu]
            valid = _valid_mask(base_mask, px, y, ctrlf)
            b = partial_slope(y, px, ctrlf, valid)
            if k == 0:
                first_b = b
            if np.isfinite(b):
                n_used += 1
                if abs(b) >= abs(beta_obs) - 1e-12:
                    count += 1
    # sanity: identity permutation reproduces the observed beta
    assert np.isfinite(first_b) and abs(first_b - beta_obs) < 1e-9, \
        f"identity perm did not reproduce beta_obs ({first_b} vs {beta_obs})"
    p_value = count / n_used if n_used else np.nan
    return beta_obs, p_value, n_edges, n_used


def bivariate_r(OUT, PRED, base_mask):
    """Raw Mantel r (no controls) on the observed edges."""
    iu = np.triu_indices(OUT.shape[0], k=1)
    y, predf = OUT[iu], PRED[iu]
    valid = _valid_mask(base_mask, predf, y, [])
    return partial_slope(y, predf, [], valid)


# ---------------------------------------------------------------------------
# conditions to run (per scheme)
# ---------------------------------------------------------------------------
def condition_specs(COS, Smat, Mmat, STR):
    ACOS = np.abs(COS)
    # (label_out, OUT, label_pred, PRED, controls)  -- controls = [STR] or []
    return [
        ("Smat", Smat, "COS", COS, [STR]),    # headline directional test
        ("Smat", Smat, "abs(COS)", ACOS, [STR]),  # sign-matters check (expect ~0)
        ("Mmat", Mmat, "COS", COS, [STR]),    # magnitude near-null sanity
        ("Smat", Smat, "COS", COS, []),       # bivariate Mantel (raw r)
        ("Mmat", Mmat, "COS", COS, []),       # bivariate Mantel (raw r)
    ]


def run_scheme(scheme, df_scheme, traits):
    COS, Smat, Mmat, STR = build_matrices(df_scheme, traits)
    base_mask = np.isfinite(utri(COS))

    # ---- sanity asserts on the matrices ----
    for name, M in [("COS", COS), ("Smat", Smat), ("Mmat", Mmat), ("STR", STR)]:
        fin = np.isfinite(M)
        assert np.allclose(M[fin], M.T[fin]), f"{name} not symmetric"
    assert int(base_mask.sum()) <= 28, "more than 28 upper-tri edges?!"
    # cross-check matrix build against pre-computed CSV columns
    idx = {t: i for i, t in enumerate(traits)}
    for _, r in df_scheme.iterrows():
        i, j = idx[r["trait_a"]], idx[r["trait_b"]]
        assert abs(Mmat[i, j] - r["mean_joint_abs"]) < 1e-6
        assert abs(STR[i, j] - r["mean_single_abs"]) < 1e-6

    rows = []
    for lo, OUT, lp, PRED, CTRLS in condition_specs(COS, Smat, Mmat, STR):
        beta, p, n_edges, n_perms = mrqap_exact(OUT, PRED, CTRLS, base_mask)
        biv = bivariate_r(OUT, PRED, base_mask)
        assert 0.0 <= p <= 1.0, "p out of [0,1]"
        rows.append({
            "outcome": lo,
            "predictor": lp,
            "controls": "[STR]" if CTRLS else "[]",
            "bivariate_r": round(biv, 4),
            "beta_std_partial": round(beta, 4),
            "exact_two_sided_p": round(p, 5),
            "n_nodes": OUT.shape[0],
            "n_perms": n_perms,
            "n_edges_used": n_edges,
        })
    return pd.DataFrame(rows), (COS, Smat, Mmat, STR, base_mask)


# ---------------------------------------------------------------------------
# fidelity check: fast path == brief's literal resid_on/std_slope on identity
# ---------------------------------------------------------------------------
def fidelity_check(COS, Smat, STR, base_mask):
    iu = np.triu_indices(COS.shape[0], k=1)
    m = base_mask
    y = utri(Smat)[m]
    zc = [utri(STR)[m]]
    ry = resid_on(y, zc)
    rx = resid_on(utri(COS)[m], zc)
    beta_literal = std_slope(rx, ry)
    beta_fast, _, _, _ = mrqap_exact(Smat, COS, [STR], base_mask)
    return beta_literal, beta_fast


# ---------------------------------------------------------------------------
# gate-removal report (raw -> gated)
# ---------------------------------------------------------------------------
def gate_report(scheme, df_raw, traits):
    sub = df_raw[df_raw["scheme"] == scheme]
    removed = sub[sub["coh_steered"] < COH_GATE][["trait_a", "trait_b", "cos", "coh_steered"]]
    removed = removed.sort_values("cos").reset_index(drop=True)
    return removed


# ---------------------------------------------------------------------------
# optional: leave-one-trait-out leverage (7x7, 7! = 5040 perms per fold)
# ---------------------------------------------------------------------------
def loo_leverage(scheme, df_scheme, traits):
    out = {}
    for lo, lp, ctrl_kind in [("Smat", "COS", "STR"), ("Smat", "abs(COS)", "STR"),
                              ("Mmat", "COS", "STR"), ("Smat", "COS", "none"),
                              ("Mmat", "COS", "none")]:
        betas, ps = [], []
        for drop in traits:
            keep = [t for t in traits if t != drop]
            sub = df_scheme[(df_scheme["trait_a"] != drop) & (df_scheme["trait_b"] != drop)]
            COS, Smat, Mmat, STR = build_matrices(sub, keep)
            base_mask = np.isfinite(utri(COS))
            OUT = Smat if lo == "Smat" else Mmat
            PRED = np.abs(COS) if lp == "abs(COS)" else COS
            CTRLS = [STR] if ctrl_kind == "STR" else []
            try:
                beta, p, n_edges, _ = mrqap_exact(OUT, PRED, CTRLS, base_mask)
            except AssertionError:
                beta, p = np.nan, np.nan
            if np.isfinite(beta):
                betas.append(beta)
                ps.append(p)
        ctrl_lbl = "[STR]" if ctrl_kind == "STR" else "[]"
        out[f"{lo}~{lp}|{ctrl_lbl}"] = (
            (min(betas), max(betas)) if betas else (np.nan, np.nan),
            (min(ps), max(ps)) if ps else (np.nan, np.nan),
            len(betas),
        )
    return out


MD_OUT = REPO / "analysis/rq1_consolidated/mrqap_significance_results.md"
CSV_OUT = REPO / "analysis/rq1_consolidated/mrqap_significance_results.csv"


def df_to_md(df):
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def write_outputs(tables, gate_removals, loo, verif):
    # combined CSV (scheme column added)
    frames = []
    for scheme, tbl in tables.items():
        t = tbl.copy()
        t.insert(0, "scheme", SCHEME_LABEL[scheme])
        frames.append(t)
    pd.concat(frames, ignore_index=True).to_csv(CSV_OUT, index=False)

    lines = []
    lines.append("# Exact MRQAP / Mantel — geometry → composition (per controlled scheme)\n")
    lines.append("Exact 8! = 40,320 node-permutation test on 8×8 trait matrices, run separately "
                 "per dose-controlled scheme. Replaces the invalid within-scheme cosine-label "
                 "shuffle over n=76 pooled pair×scheme rows (which ignored dyadic dependence and "
                 "double-counted each pair's cosine across schemes).\n")
    lines.append(f"- **Data**: `{CSV_GATED.relative_to(REPO)}` (coherence gate "
                 f"`coh_steered >= {COH_GATE:g}` already applied); gate removals confirmed against "
                 f"`{CSV_RAW.relative_to(REPO)}`.\n")
    lines.append(f"- **Predictor**: signed cosine of unit steering directions (identical across "
                 f"schemes; max |Δcos| = {verif['cos_dmax']:.1e}).\n")
    lines.append(f"- **Outcomes**: `Smat` (directional, signed reinforce−suppress) and `Mmat` "
                 f"(joint magnitude). **Control**: `STR` (individual strength). Mechanical-push "
                 f"control omitted (per_axis: constant by construction; normalized_sum: deterministic "
                 f"function of cosine — would absorb the signal).\n")
    lines.append("- **NaN under relabeling**: gated dyads are NaN in every matrix; available-case "
                 "node permutation (statistic on commonly-observed dyads, controls refit each time). "
                 "Reduces exactly to the fixed-mask formula when no dyad is gated (normalized_sum).\n")
    lines.append(f"- `beta_std_partial` = standardized partial slope = partial correlation; "
                 f"`bivariate_r` = raw Mantel r (no control).\n")

    lines.append("\n## Coherence-gate removals\n")
    for scheme in CONTROLLED:
        rem = gate_removals[scheme]
        if len(rem) == 0:
            lines.append(f"- **{SCHEME_LABEL[scheme]}**: removed 0 dyads (all 28 pairs survive; "
                         f"effective n = 28 edges).")
        else:
            lines.append(f"- **{SCHEME_LABEL[scheme]}**: removed {len(rem)} dyads "
                         f"(effective n = {28 - len(rem)} edges). Gated pairs (coh<{COH_GATE:g}):")
            for _, r in rem.iterrows():
                lines.append(f"  - {r['trait_a']}–{r['trait_b']} (cos={r['cos']:+.3f}, "
                             f"coh={r['coh_steered']:.1f})")
            n_opp = int((rem["cos"] <= 0).sum())
            lines.append(f"  - {n_opp}/{len(rem)} gated dyads have cos ≤ 0: per_axis loses most of "
                         f"the opposed side, so its directional estimate leans aligned; the "
                         f"suppression side rests on normalized_sum (which retains all opposed pairs).")

    for scheme in CONTROLLED:
        lines.append(f"\n## Scheme: {SCHEME_LABEL[scheme]}  (n_edges = "
                     f"{int(tables[scheme]['n_edges_used'].iloc[0])}, n_perms = 40320)\n")
        lines.append(df_to_md(tables[scheme]))

    lines.append("\n## Leave-one-trait-out leverage (7×7, 5040 perms/fold)\n")
    for scheme in CONTROLLED:
        lines.append(f"\n**{SCHEME_LABEL[scheme]}** — range of (β_std_partial, p) across 8 folds:\n")
        lines.append("| condition | β range | p range | folds |")
        lines.append("|---|---|---|---|")
        for cond, ((blo, bhi), (plo, phi), nfold) in loo[scheme].items():
            cond_md = cond.replace("|", "\\|")  # escape pipe so the md table column survives
            lines.append(f"| {cond_md} | [{blo:+.3f}, {bhi:+.3f}] | [{plo:.4f}, {phi:.4f}] | {nfold}/8 |")

    MD_OUT.write_text("\n".join(lines) + "\n")
    print(f"\n[written] {MD_OUT.relative_to(REPO)}")
    print(f"[written] {CSV_OUT.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    gated, raw, traits = load_frames()
    print("=" * 78)
    print("EXACT MRQAP / MANTEL  —  geometry -> composition, per controlled scheme")
    print("=" * 78)
    print(f"gated frame : {CSV_GATED.relative_to(REPO)}")
    print(f"raw frame   : {CSV_RAW.relative_to(REPO)}")
    print(f"coherence gate: coh_steered >= {COH_GATE:g}  (already applied to the gated frame)")
    print(f"traits (n={len(traits)}): {traits}")
    print(f"scheme label map: {SCHEME_LABEL}")
    assert len(traits) == 8, "expected 8 traits"

    # cross-scheme verification: cosine shared, single-injection deltas scheme-independent
    print("\n--- cross-scheme verification (single-injection settings + cosine) ---")
    piv_cos = gated.pivot_table(index="pair_key", columns="scheme", values="cos")
    dmax = float(np.nanmax(np.abs(piv_cos["normTrue"] - piv_cos["per_axis"])))
    verif = {"cos_dmax": dmax}
    print(f"max |cos(normTrue) - cos(per_axis)| over shared pairs: {dmax:.2e}  (expect 0 -> shared)")
    for col, lbl in [("delta_a_single", "single_a"), ("delta_b_single", "single_b")]:
        piv = gated.pivot_table(index="pair_key", columns="scheme", values=col)
        both = piv.dropna()
        diff = (both["normTrue"] - both["per_axis"]).abs()
        r = np.corrcoef(both["normTrue"], both["per_axis"])[0, 1]
        print(f"{lbl}: shared pairs={len(both)}  r(normTrue,per_axis)={r:.3f}  "
              f"max|diff|={diff.max():.2f}  mean|diff|={diff.mean():.2f}  "
              f"(nominally identical; separately sampled -> small noise)")

    # gate removals per scheme
    print("\n--- coherence-gate removals (raw -> gated), per scheme ---")
    gate_removals = {}
    for scheme in CONTROLLED:
        rem = gate_report(scheme, raw, traits)
        gate_removals[scheme] = rem
        lbl = SCHEME_LABEL[scheme]
        if len(rem) == 0:
            print(f"[{lbl}] removed 0 dyads (all 28 pairs survive)")
        else:
            print(f"[{lbl}] removed {len(rem)} dyads (coh_steered < {COH_GATE:g}); "
                  f"cos of removed: {[round(c, 3) for c in rem['cos']]}")
            for _, r in rem.iterrows():
                print(f"      {r['trait_a']:>13s}--{r['trait_b']:<13s}  cos={r['cos']:+.4f}  coh={r['coh_steered']:.2f}")

    # per-scheme MRQAP tables
    store = {}
    tables = {}
    for scheme in CONTROLLED:
        df_s = gated[gated["scheme"] == scheme].reset_index(drop=True)
        tbl, mats = run_scheme(scheme, df_s, traits)
        store[scheme] = (tbl, mats)
        tables[scheme] = tbl
        lbl = SCHEME_LABEL[scheme]
        print("\n" + "=" * 78)
        print(f"SCHEME: {lbl}   (data scheme='{scheme}', n_pairs={len(df_s)})")
        print("=" * 78)
        print(tbl.to_string(index=False))

    # fidelity assert: fast partial slope == brief's literal resid_on/std_slope (normTrue, full mask)
    COS, Smat, Mmat, STR, base_mask = store["normTrue"][1]
    bl, bf = fidelity_check(COS, Smat, STR, base_mask)
    print(f"\n[fidelity] normTrue Smat~COS|[STR]: literal std_slope={bl:+.6f}  "
          f"fast partial_slope={bf:+.6f}  diff={abs(bl - bf):.2e}")
    assert abs(bl - bf) < 1e-9, "fast path disagrees with brief's literal formula"
    print("[sanity] all asserts passed: matrices symmetric; n_edges<=28; "
          "identity perm reproduces beta_obs; 0<=p<=1; fast==literal.")

    # optional leave-one-trait-out leverage
    print("\n" + "=" * 78)
    print("OPTIONAL — leave-one-trait-out leverage (7x7, 7! = 5040 perms per fold)")
    print("=" * 78)
    loo = {}
    for scheme in CONTROLLED:
        df_s = gated[gated["scheme"] == scheme].reset_index(drop=True)
        lev = loo_leverage(scheme, df_s, traits)
        loo[scheme] = lev
        print(f"\n[{SCHEME_LABEL[scheme]}]  (range of beta_std_partial and p across the 8 folds)")
        for cond, ((blo, bhi), (plo, phi), nfold) in lev.items():
            print(f"   {cond:18s}  beta in [{blo:+.3f}, {bhi:+.3f}]   "
                  f"p in [{plo:.4f}, {phi:.4f}]   ({nfold}/8 folds estimable)")

    write_outputs(tables, gate_removals, loo, verif)


if __name__ == "__main__":
    main()
