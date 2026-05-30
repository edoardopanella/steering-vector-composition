import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score


def _chi2_stat(b, r, n_bin=None, n_reg=None):
    if n_bin is None:
        n_bin = b.max() + 1
    if n_reg is None:
        n_reg = r.max() + 1
    t = np.zeros((n_bin, n_reg))
    np.add.at(t, (b, r), 1)
    exp = t.sum(1, keepdims=True) @ t.sum(0, keepdims=True) / t.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nansum((t - exp) ** 2 / exp)


def bootstrap_perm_multi(model, X, y, exp_auc, N_BOOT, N_PERM, rng):
    classes = sorted(np.unique(y))
    n_obs   = len(y)
    n_feat  = X.shape[1]

    def fit_auc_multi(X_tr, y_tr, X_ev, y_ev):
        if len(np.unique(y_tr)) < len(classes):
            return None
        m = clone(model).fit(X_tr, y_tr)
        if len(m.classes_) < len(classes):
            return None
        return roc_auc_score(y_ev, m.predict_proba(X_ev), multi_class='ovr', average='macro')

    loo_slopes      = []
    loo_degen_folds = 0
    for i in range(n_obs):
        mask = np.ones(n_obs, dtype=bool)
        mask[i] = False
        y_tr = y[mask]
        if len(np.unique(y_tr)) < len(classes):
            loo_degen_folds += 1
            continue
        try:
            m = clone(model).fit(X[mask], y_tr)
            loo_slopes.append(np.abs(m.coef_).mean(axis=0).tolist())
        except Exception:
            pass

    loo_slopes_arr = np.array(loo_slopes)
    print(f"LOO folds with missing class in training: {loo_degen_folds}")
    for feat_idx in range(n_feat):
        col = loo_slopes_arr[:, feat_idx]
        print(f"  feature {feat_idx}: mean|slope| min={col.min():.3f}  max={col.max():.3f}")

    boot_aucs = []
    for _ in range(N_BOOT):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        try:
            v = fit_auc_multi(X[idx], y[idx], X[idx], y[idx])
            if v is not None:
                boot_aucs.append(v)
        except Exception:
            pass

    boot_aucs = np.array(boot_aucs)
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"AUC = {exp_auc:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  "
          f"(n_boot={len(boot_aucs)}  shape: min={boot_aucs.min():.3f} "
          f"median={np.median(boot_aucs):.3f} max={boot_aucs.max():.3f})")

    perm_aucs = []
    for _ in range(N_PERM):
        y_perm = rng.permutation(y)
        try:
            v = fit_auc_multi(X, y_perm, X, y_perm)
            if v is not None:
                perm_aucs.append(v)
        except Exception:
            pass

    perm_aucs = np.array(perm_aucs)
    p_value   = (np.sum(perm_aucs >= exp_auc) + 1) / (N_PERM + 1)
    print(f"permutation p (one-sided) = {p_value:.4f}")

    return {
        "auc_insample":    exp_auc,
        "ci_lo":           float(ci_lo),
        "ci_hi":           float(ci_hi),
        "n_boot":          len(boot_aucs),
        "boot_aucs":       boot_aucs.tolist(),
        "perm_p":          float(p_value),
        "perm_auc_mean":   float(perm_aucs.mean()),
        "loo_degen_folds": loo_degen_folds,
        "loo_n_folds":     len(loo_slopes),
    }


def bootstrap_perm(model, X, y, exp_auc, N_BOOT, N_PERM, rng):

    def fit_auc(X_tr, y_tr, X_ev, y_ev):
        m = clone(model).fit(X_tr, y_tr)
        return roc_auc_score(y_ev, m.predict_proba(X_ev)[:, 1])

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_obs   = len(y)

    # --- LOO slope stability ---
    # For each leave-one-out fold: fit on n-1 points, extract the slope
    # coefficient (coef_[0, :]), track sign and range. Folds where the training
    # set has < 2 positives are skipped and counted (a sanity gate for setups
    # with very few positives).
    loo_slopes        = []
    loo_degen_folds   = 0
    for i in range(n_obs):
        mask   = np.ones(n_obs, dtype=bool)
        mask[i] = False
        y_tr   = y[mask]
        if (y_tr == 1).sum() < 2:
            loo_degen_folds += 1
            continue
        try:
            m = clone(model).fit(X[mask], y_tr)
            loo_slopes.append(m.coef_[0].tolist())    # one value per feature
        except Exception:
            pass

    loo_slopes_arr = np.array(loo_slopes)              # shape: (n_valid_folds, n_features)
    print(f"LOO folds with < 2 positives in training: {loo_degen_folds}")
    for feat_idx in range(loo_slopes_arr.shape[1]):
        col = loo_slopes_arr[:, feat_idx]
        print(f"  feature {feat_idx}: slope min={col.min():+.3f}  "
              f"max={col.max():+.3f}  "
              f"n_negative={int((col < 0).sum())}/{len(col)}")

    # --- Bootstrap 95% CI (stratified, evaluate on resample) ---
    boot_aucs = []
    for _ in range(N_BOOT):
        pi  = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ni  = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pi, ni])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boot_aucs.append(fit_auc(X[idx], y[idx], X[idx], y[idx]))
        except ValueError:
            pass

    boot_aucs = np.array(boot_aucs)
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"AUC = {exp_auc:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  "
          f"(n_boot={len(boot_aucs)}  shape: min={boot_aucs.min():.3f} "
          f"median={np.median(boot_aucs):.3f} max={boot_aucs.max():.3f})")

    # --- Permutation p-value (one-sided) ---
    perm_aucs = []
    for _ in range(N_PERM):
        y_perm = rng.permutation(y)
        perm_aucs.append(fit_auc(X, y_perm, X, y_perm))

    perm_aucs = np.array(perm_aucs)
    p_value   = (np.sum(perm_aucs >= exp_auc) + 1) / (N_PERM + 1)
    print(f"permutation p (one-sided) = {p_value:.4f}")

    return {
        "auc_insample":           exp_auc,
        "ci_lo":                  float(ci_lo),
        "ci_hi":                  float(ci_hi),
        "n_boot":                 len(boot_aucs),
        "boot_aucs":              boot_aucs.tolist(),      # full distribution for figures
        "perm_p":                 float(p_value),
        "perm_auc_mean":          float(perm_aucs.mean()),
        "loo_slope_min":          float(loo_slopes_arr[:, 0].min()),
        "loo_slope_max":          float(loo_slopes_arr[:, 0].max()),
        "loo_slope_n_neg":        int((loo_slopes_arr[:, 0] < 0).sum()),
        "loo_slope_n_folds":      len(loo_slopes),
        "loo_degen_folds":        loo_degen_folds,
    }
