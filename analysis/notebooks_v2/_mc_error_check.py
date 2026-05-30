"""
One-off: is n_boot=2000 / n_perm=10000 enough?

Two independent Monte-Carlo (MC) quantities:
  - perm_p  : (#{perm_auc >= obs} + 1)/(n_perm+1)  -> binomial-tail estimator
  - 95% CI  : [2.5, 97.5] percentiles of n_boot resampled AUCs

For each we compare the production setting against a high-resolution reference
and report the MC scatter you would actually see at the production setting.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from bins_analysis_scripts import DATASETS, DEFAULT_SEM, add_semantic, load_data

PROD_BOOT, PROD_PERM = 2000, 10000
REF_BOOT, REF_PERM = 40000, 100000   # high-resolution reference
N_REPLICATES = 400                   # subsamples used to measure scatter

BIN_SPECS = [
    ("B1_abs_cos", ["cosine_abs"]),
    ("B2_abs_cos_sem", ["cosine_abs", "sem_sim"]),
    ("B3_signed_cos", ["cos"]),
    ("B4_signed_cos_sem", ["cos", "sem_sim"]),
]


def ref_distributions(X, y, rng):
    """Big reference null (perm) and bootstrap AUC distributions for one spec."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    model = LogisticRegression(penalty=None)

    def auc(Xt, yt):
        m = model.fit(Xt, yt)
        return roc_auc_score(yt, m.predict_proba(Xt)[:, 1])

    obs = auc(X, y)

    perm = np.fromiter((auc(X, rng.permutation(y)) for _ in range(REF_PERM)),
                       float, REF_PERM)

    boot = []
    for _ in range(REF_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos), True),
                              rng.choice(neg, len(neg), True)])
        if len(np.unique(y[idx])) == 2:
            boot.append(auc(X[idx], y[idx]))
    return obs, perm, np.array(boot)


def perm_p(null, obs, n):
    """p-value estimate from a size-n subsample of the reference null."""
    s = rng_sub.choice(null, n, replace=False)
    return (np.sum(s >= obs) + 1) / (n + 1)


print("Loading norm dataset...")
df, _ = load_data(DATASETS["norm"]["summary"])
df = add_semantic(df, DEFAULT_SEM)
df["is_additive"] = (df["regime"] == "additive").astype(int)
scaler = StandardScaler()

rng = np.random.default_rng(12345)     # builds reference distributions
rng_sub = np.random.default_rng(999)   # draws subsamples for scatter

print(f"\nReference: {REF_PERM:,} perms / {REF_BOOT:,} boots per experiment")
print(f"Production: {PROD_PERM:,} perms / {PROD_BOOT:,} boots\n")

print("=" * 92)
print("PERMUTATION p-value  (decision threshold = 0.05)")
print("=" * 92)
print(f"{'exp':<18}{'p_ref':>9}{'SE@10k':>9}{'p_prod mean':>13}"
      f"{'p_prod sd':>11}{'p_prod range':>20}{'cross .05?':>12}")
for name, cols in BIN_SPECS:
    X = scaler.fit_transform(df[cols].to_numpy())
    y = df["is_additive"].to_numpy()
    obs, perm_ref, boot_ref = ref_distributions(X, y, rng)

    p_ref = (np.sum(perm_ref >= obs) + 1) / (REF_PERM + 1)
    se = np.sqrt(p_ref * (1 - p_ref) / PROD_PERM)               # binomial SE
    ps = np.array([perm_p(perm_ref, obs, PROD_PERM) for _ in range(N_REPLICATES)])
    crosses = (ps.min() < 0.05) and (ps.max() >= 0.05)
    print(f"{name:<18}{p_ref:>9.4f}{se:>9.4f}{ps.mean():>13.4f}"
          f"{ps.std():>11.4f}{f'[{ps.min():.4f}, {ps.max():.4f}]':>20}"
          f"{('YES' if crosses else 'no'):>12}")

    # stash boot ref for the next block (recompute cheap things only)
    BIN_SPECS_BOOT = BIN_SPECS  # noqa
    globals().setdefault("_boot", {})[name] = (obs, boot_ref)

print("\n" + "=" * 92)
print("BOOTSTRAP 95% CI endpoints")
print("=" * 92)
print(f"{'exp':<18}{'ci_lo_ref':>11}{'ci_hi_ref':>11}"
      f"{'lo sd@2k':>11}{'hi sd@2k':>11}{'lo range@2k':>20}")
for name, _ in BIN_SPECS:
    obs, boot_ref = globals()["_boot"][name]
    lo_ref, hi_ref = np.percentile(boot_ref, [2.5, 97.5])
    los, his = [], []
    for _ in range(N_REPLICATES):
        s = rng_sub.choice(boot_ref, PROD_BOOT, replace=False)
        lo, hi = np.percentile(s, [2.5, 97.5])
        los.append(lo); his.append(hi)
    los, his = np.array(los), np.array(his)
    print(f"{name:<18}{lo_ref:>11.4f}{hi_ref:>11.4f}"
          f"{los.std():>11.4f}{his.std():>11.4f}"
          f"{f'[{los.min():.3f}, {los.max():.3f}]':>20}")

print("\n" + "=" * 92)
print("ANALYTIC binomial SE of perm_p at n_perm=10000, by true p")
print("=" * 92)
for p in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
    se = np.sqrt(p * (1 - p) / PROD_PERM)
    print(f"  true p={p:<6}  SE={se:.5f}   ~95% MC interval [{max(0,p-2*se):.4f}, {p+2*se:.4f}]")
