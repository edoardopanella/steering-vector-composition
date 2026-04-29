import numpy as np
import pandas as pd

def get_offdiag(G: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    n = G.shape[0]
    idx = np.triu_indices(n, k=1)
    return G[idx], idx

def make_pairs_df(G: np.ndarray, behavior_order: list[str]) -> pd.DataFrame:
    values, idx = get_offdiag(G)
    rows, cols = idx
    return pd.DataFrame({
        "i": rows,
        "j": cols,
        "behavior_i": [behavior_order[i] for i in rows],
        "behavior_j": [behavior_order[j] for j in cols],
        "cosine": values,
        "abs_cosine": np.abs(values),
    })

def stratify_pairs(
    pairs_df: pd.DataFrame,
    n_near: int = 14,
    n_moderate: int = 13,
    n_high: int = 13,
    seed: int = 42,
) -> pd.DataFrame:
    
    near     = pairs_df[pairs_df["abs_cosine"] < 0.2]
    moderate = pairs_df[(pairs_df["abs_cosine"] >= 0.2) & (pairs_df["abs_cosine"] < 0.35)]
    high     = pairs_df[pairs_df["abs_cosine"] >= 0.35]  
    
    def _sample(bin_df, n_target, label):
        if len(bin_df) < n_target:
            print(f"[WARN] stratum '{label}' has {len(bin_df)} pairs, wanted {n_target}")
            n_target = len(bin_df)
        return bin_df.sample(n=n_target, random_state=seed)
    
    near_sample     = _sample(near, n_near, "near").assign(stratum="near")
    moderate_sample = _sample(moderate, n_moderate, "moderate").assign(stratum="moderate")
    high_sample     = _sample(high, n_high, "high").assign(stratum="high")

    return pd.concat([near_sample, moderate_sample, high_sample]).reset_index(drop=True)
    