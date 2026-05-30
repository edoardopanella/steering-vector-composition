"""
Validation of the exact MRQAP p-values. Exercises the ACTUAL mrqap_exact() used for the
reported numbers, and checks five things:

  1. Permutation set is complete and unique (exactly 8! = 40,320 distinct relabelings).
  2. Observed beta == an INDEPENDENT recomputation (scipy.pearsonr on numpy lstsq residuals,
     a code path that shares nothing with mrqap_exact's internals).
  3. Determinism: same inputs -> identical p on re-run (no RNG in the test itself).
  4. Known-answer: predictor == outcome -> p is the floor (perfect association);
     predictor == -outcome -> same p (two-sided).
  5. Calibration / type-I error: under the null (predictor independent of the outcome) the
     exact p must be ~Uniform(0,1). We feed many random null predictors through the SAME
     function and check mean ~ 0.5, frac<=0.05 ~ 0.05, KS vs uniform. Done for BOTH normTrue
     (full 28-edge mask) and per_axis (22-edge gated mask, available-case path) -- the latter
     is the assumption-laden one, so calibrating it is the key evidence the p=0.051 is honest.

Run:  ./venv/bin/python analysis/rq1_mrqap_validate.py
"""
from itertools import permutations

import numpy as np
from scipy import stats

from rq1_mrqap_significance import (
    load_frames, build_matrices, utri, mrqap_exact, SCHEME_LABEL, CONTROLLED,
)


def indep_partial_corr(OUT, PRED, CTRL, valid):
    """Partial correlation of OUT,PRED given CTRL on `valid` edges -- fully independent of
    mrqap_exact (scipy.pearsonr on lstsq residuals)."""
    iu = np.triu_indices(OUT.shape[0], 1)
    y, x, z = OUT[iu][valid], PRED[iu][valid], CTRL[iu][valid]
    Z = np.column_stack([np.ones_like(z), z])
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    return stats.pearsonr(rx, ry)[0]


def rand_sym_like(template, rng, structured):
    """Random symmetric matrix with the SAME NaN (gated + diagonal) pattern as `template`."""
    n = template.shape[0]
    M = np.full((n, n), np.nan)
    s = rng.normal(size=n)
    for i in range(n):
        for j in range(i + 1, n):
            v = (s[i] + s[j] + 0.1 * rng.normal()) if structured else rng.normal()
            M[i, j] = M[j, i] = v
    M[~np.isfinite(template)] = np.nan   # copy gate + diagonal mask
    return M


def calibrate(OUT, COS, STR, base_mask, rng, k, structured):
    ps = np.empty(k)
    for t in range(k):
        P = rand_sym_like(COS, rng, structured)
        _, p, _, _ = mrqap_exact(OUT, P, [STR], base_mask)
        ps[t] = p
    return ps


def main():
    gated, _, traits = load_frames()
    print("=" * 74)
    print("VALIDATION OF THE EXACT MRQAP p-VALUES")
    print("=" * 74)

    # ---- 1. permutation group complete + unique ----
    perms = list(permutations(range(8)))
    assert len(perms) == 40320 and len(set(perms)) == 40320
    print(f"\n[1] permutation group: {len(perms)} perms, all distinct, identity first = "
          f"{perms[0] == tuple(range(8))}  -> OK")

    mats = {}
    for scheme in CONTROLLED:
        df_s = gated[gated["scheme"] == scheme].reset_index(drop=True)
        COS, Smat, Mmat, STR = build_matrices(df_s, traits)
        base_mask = np.isfinite(utri(COS))
        mats[scheme] = (COS, Smat, Mmat, STR, base_mask)

    # ---- 2. observed beta vs independent recompute ----
    print("\n[2] observed beta_obs  vs  independent scipy/lstsq partial correlation:")
    for scheme in CONTROLLED:
        COS, Smat, Mmat, STR, base_mask = mats[scheme]
        beta_obs, p, n_edges, _ = mrqap_exact(Smat, COS, [STR], base_mask)
        valid = base_mask & np.isfinite(utri(COS)) & np.isfinite(utri(Smat)) & np.isfinite(utri(STR))
        beta_ind = indep_partial_corr(Smat, COS, STR, valid)
        print(f"    {SCHEME_LABEL[scheme]:14s}  mrqap={beta_obs:+.6f}  independent={beta_ind:+.6f}  "
              f"|diff|={abs(beta_obs - beta_ind):.2e}  (p={p:.5f}, edges={n_edges})")
        assert abs(beta_obs - beta_ind) < 1e-9, "observed beta disagrees with independent recompute"

    # ---- 3. determinism ----
    print("\n[3] determinism (same inputs, two calls):")
    for scheme in CONTROLLED:
        COS, Smat, Mmat, STR, base_mask = mats[scheme]
        _, p1, _, _ = mrqap_exact(Smat, COS, [STR], base_mask)
        _, p2, _, _ = mrqap_exact(Smat, COS, [STR], base_mask)
        print(f"    {SCHEME_LABEL[scheme]:14s}  p1={p1:.6f}  p2={p2:.6f}  identical={p1 == p2}")
        assert p1 == p2

    # ---- 4. known-answer: predictor == outcome ----
    print("\n[4] known-answer (predictor = outcome -> perfect assoc -> p at the floor):")
    for scheme in CONTROLLED:
        COS, Smat, Mmat, STR, base_mask = mats[scheme]
        b_self, p_self, _, nperm = mrqap_exact(Smat, Smat, [STR], base_mask)
        b_neg, p_neg, _, _ = mrqap_exact(Smat, -Smat, [STR], base_mask)
        print(f"    {SCHEME_LABEL[scheme]:14s}  beta(self)={b_self:+.4f}  p(self)={p_self:.6f}  "
              f"p(-self)={p_neg:.6f}  (two-sided -> equal: {abs(p_self - p_neg) < 1e-12})")
        assert b_self > 0.999 and p_self <= 5.0 / nperm + 1e-9 and abs(p_self - p_neg) < 1e-9

    # ---- 5. calibration under the null ----
    print("\n[5] calibration: random NULL predictors -> p must be ~Uniform(0,1)")
    print("    (mean~0.50, frac<=0.05 ~0.05, frac<=0.10 ~0.10, KS p large = consistent w/ uniform)")
    rng = np.random.default_rng(12345)
    # (structured?, K) per scheme; per_axis is ~4x slower per enum so fewer reps
    plan = {"normTrue": [(False, 250), (True, 150)], "per_axis": [(False, 130)]}
    for scheme in CONTROLLED:
        COS, Smat, Mmat, STR, base_mask = mats[scheme]
        for structured, k in plan[scheme]:
            lbl = "node-structured" if structured else "iid"
            ps = calibrate(Smat, COS, STR, base_mask, rng, k, structured)
            ks = stats.kstest(ps, "uniform")
            print(f"    {SCHEME_LABEL[scheme]:14s} [{lbl:15s}] K={len(ps):4d}  "
                  f"mean={ps.mean():.3f}  frac<=.05={np.mean(ps <= 0.05):.3f}  "
                  f"frac<=.10={np.mean(ps <= 0.10):.3f}  KS_p={ks.pvalue:.3f}", flush=True)

    print("\n" + "=" * 74)
    print("All structural asserts passed. See [5] for empirical type-I calibration.")
    print("=" * 74)


if __name__ == "__main__":
    main()
