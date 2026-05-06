"""
RQ2 Phase 1 — tau recalibration on existing pilot trajectories.

Original recipe (calibrate_tau in src/joint_analysis/joint_injection.py) used
1.5 x max layer-to-layer step in the raw individual-steering projection. That
overshot because raw projections grow with residual-stream norm (E10.2:
||h(L)|| roughly monotone in L), so a single-layer step in raw pi is O(alpha *
||h||) — much larger than the mean eq (2) integrand ever gets. Result: tau =
9.87, all L_div saturated at L_final.

Goal: tau should be a noise floor on the eq (2) integrand
|pi^(1,1)(L) - pi^(indiv)(L)| under the null hypothesis "joint = individual"
(i.e. additive composition). Read off from the pilot's individual-steering
data, then a 1.5x cushion.

Three candidate recipes, all measured on the same pilot artifacts, no
recompute of the model:

(R1) Pooled inter-prompt std of individual trajectories:
    tau = 1.5 * max_{L, pair, direction} std_across_prompts(pi^(indiv)(L))
    Captures cross-prompt natural variation. Crude but transparent.

(R2) Split-half null bootstrap of the integrand:
    For each (pair, individual setting, direction), randomly split N samples
    into two halves of N/2, compute |mean_A(L) - mean_B(L)|, track max over L.
    Repeat B times. Take 95th percentile of the resulting distribution.
    tau = 1.5 * 95th percentile across all (pair, direction) seeds.
    Mirrors what the actual integrand measurement does (mean over prompts of
    |Delta pi|), under the null.

(R3) Same-prompt inter-completion variance (cleanest, but small N):
    Per prompt we have N_COMPLETIONS_PER_PROMPT=3 completions under each
    individual setting. Pairwise |pi_a(L) - pi_b(L)| across these gives a
    direct null distribution at fixed prompt. Pool across all (prompt, pair,
    direction), take 95th percentile.
    tau = 1.5 * 95th percentile.

Outputs:
    - prints tau under each recipe
    - prints L_div median per pair under each recipe
    - writes results/anthropic_repl/trajectory_pilot_l17/Llama-3.1-8B-Instruct/
      tau_recalibration.json with all numbers, for pre-registration
    - writes analysis/figures/fig_tau_recalibration.png comparing the three
      taus against the eq (2) integrand curves

No model load. Reads projections.pt + pilot_summary.json, recomputes locally.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

OUT_DIR = Path("results/anthropic_repl/trajectory_pilot_l17/Llama-3.1-8B-Instruct")
FIG_DIR = Path("analysis/figures/trajectory_pilot")
SUMMARY_PATH = OUT_DIR / "pilot_summary.json"
RECAL_OUT = OUT_DIR / "tau_recalibration.json"

TAU_FACTOR = 1.5
BOOTSTRAP_B = 1000
BOOTSTRAP_PERCENTILE = 95.0
SEED = 0


# === loaders ===============================================================

def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text())


def load_pair_projections(pair: tuple[str, str]) -> dict:
    p = OUT_DIR / f"{pair[0]}__{pair[1]}" / "projections.pt"
    blob = torch.load(p, map_location="cpu", weights_only=False)
    out: dict[tuple[int, int], dict[str, dict[int, torch.Tensor]]] = {}
    for s_key, dirs in blob["projections"].items():
        ai, aj = (int(x) for x in s_key.split("_"))
        out[(ai, aj)] = {
            key: {int(L): t for L, t in d.items()}
            for key, d in dirs.items()
        }
    return {"cos": blob["cos"], "projections": out}


# === tau recipes ===========================================================

def tau_R1(pair_blobs: dict) -> tuple[float, dict]:
    """Pooled inter-prompt std under individual steering."""
    raw_std_max = 0.0
    detail = {}
    for pair, blob in pair_blobs.items():
        proj = blob["projections"]
        for setting, direction in [((1, 0), "pi_i"), ((0, 1), "pi_j")]:
            traj = proj[setting][direction]
            stds = {L: traj[L].std(unbiased=False).item() for L in traj}
            mx = max(stds.values())
            detail[f"{pair[0]}+{pair[1]}|{setting}|{direction}"] = round(mx, 4)
            if mx > raw_std_max:
                raw_std_max = mx
    return TAU_FACTOR * raw_std_max, detail


def tau_R2(pair_blobs: dict, B: int = BOOTSTRAP_B, pct: float = BOOTSTRAP_PERCENTILE) -> tuple[float, dict]:
    """Split-half null bootstrap of the integrand."""
    g = torch.Generator().manual_seed(SEED)
    pooled_max_per_boot: list[float] = []
    detail = {}
    for pair, blob in pair_blobs.items():
        proj = blob["projections"]
        for setting, direction in [((1, 0), "pi_i"), ((0, 1), "pi_j")]:
            traj = proj[setting][direction]
            layers = sorted(traj.keys())
            stacked = torch.stack([traj[L] for L in layers], dim=0)  # [n_L, N]
            n_L, N = stacked.shape
            half = N // 2
            seed_max = []
            for _ in range(B):
                perm = torch.randperm(N, generator=g)
                A_idx, B_idx = perm[:half], perm[half:half * 2]
                mean_A = stacked[:, A_idx].mean(dim=1)
                mean_B = stacked[:, B_idx].mean(dim=1)
                gap_per_layer = (mean_A - mean_B).abs()
                seed_max.append(gap_per_layer.max().item())
            tens = torch.tensor(seed_max)
            q = tens.quantile(pct / 100.0).item()
            detail[f"{pair[0]}+{pair[1]}|{setting}|{direction}"] = round(q, 4)
            pooled_max_per_boot.extend(seed_max)
    pooled = torch.tensor(pooled_max_per_boot)
    overall_q = pooled.quantile(pct / 100.0).item()
    return TAU_FACTOR * overall_q, detail


def tau_R3(pair_blobs: dict, n_completions_per_prompt: int, pct: float = BOOTSTRAP_PERCENTILE) -> tuple[float, dict]:
    """Same-prompt inter-completion variance.

    Trajectories are flat-stacked as [N = n_prompts * n_completions]; pairs
    sharing a prompt are at indices [n_completions*p, n_completions*p + 1, ...].
    """
    pooled_diffs: list[float] = []
    detail = {}
    nc = n_completions_per_prompt
    for pair, blob in pair_blobs.items():
        proj = blob["projections"]
        for setting, direction in [((1, 0), "pi_i"), ((0, 1), "pi_j")]:
            traj = proj[setting][direction]
            layers = sorted(traj.keys())
            stacked = torch.stack([traj[L] for L in layers], dim=0)  # [n_L, N]
            N = stacked.shape[1]
            n_prompts = N // nc
            pair_diffs = []
            for p in range(n_prompts):
                idx = list(range(p * nc, (p + 1) * nc))
                for i in range(len(idx)):
                    for j in range(i + 1, len(idx)):
                        diff = (stacked[:, idx[i]] - stacked[:, idx[j]]).abs()
                        pair_diffs.append(diff.max().item())
            tens = torch.tensor(pair_diffs)
            q = tens.quantile(pct / 100.0).item()
            detail[f"{pair[0]}+{pair[1]}|{setting}|{direction}"] = round(q, 4)
            pooled_diffs.extend(pair_diffs)
    pooled = torch.tensor(pooled_diffs)
    overall_q = pooled.quantile(pct / 100.0).item()
    return TAU_FACTOR * overall_q, detail


# === L_div under a given tau ===============================================

def first_crossing(integrand: dict[int, torch.Tensor], tau: float, layer_final: int) -> torch.Tensor:
    layers = sorted(integrand.keys())
    diffs = torch.stack([integrand[L] for L in layers], dim=0)  # [n_L, N]
    mask = diffs > tau
    any_cross = mask.any(dim=0)
    first_idx = mask.long().argmax(dim=0)
    layer_tensor = torch.tensor(layers, dtype=torch.long)
    picked = layer_tensor[first_idx]
    return torch.where(any_cross, picked, torch.full_like(picked, layer_final))


def integrand_per_pair(pair_blobs: dict) -> dict:
    """Returns {pair: {"i": dict[L]->[N], "j": dict[L]->[N], "layers": [..]}}.
    The eq (2) integrand at the per-prompt level (not yet meaned)."""
    out = {}
    for pair, blob in pair_blobs.items():
        proj = blob["projections"]
        layers = sorted(proj[(1, 0)]["pi_i"].keys())
        diff_i = {L: (proj[(1, 1)]["pi_i"][L] - proj[(1, 0)]["pi_i"][L]).abs() for L in layers}
        diff_j = {L: (proj[(1, 1)]["pi_j"][L] - proj[(0, 1)]["pi_j"][L]).abs() for L in layers}
        out[pair] = {"i": diff_i, "j": diff_j, "layers": layers}
    return out


def L_div_table(integrands: dict, tau: float) -> dict:
    rows = []
    for pair, info in integrands.items():
        layer_final = info["layers"][-1]
        L_i = first_crossing(info["i"], tau, layer_final)
        L_j = first_crossing(info["j"], tau, layer_final)
        rows.append({
            "pair": list(pair),
            "L_div_i_median": int(L_i.median().item()),
            "L_div_j_median": int(L_j.median().item()),
            "L_div_i_min": int(L_i.min().item()),
            "L_div_j_min": int(L_j.min().item()),
            "frac_crossed_i": float((L_i < layer_final).float().mean().item()),
            "frac_crossed_j": float((L_j < layer_final).float().mean().item()),
        })
    return rows


# === plot ==================================================================

def plot_taus(integrands: dict, taus: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    pairs = list(integrands.keys())
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4.2), sharey=True)
    if len(pairs) == 1:
        axes = [axes]

    colours = {"R1": "tab:red", "R2": "tab:green", "R3": "tab:purple"}

    for ax, pair in zip(axes, pairs):
        info = integrands[pair]
        layers = info["layers"]
        mean_i = [info["i"][L].mean().item() for L in layers]
        mean_j = [info["j"][L].mean().item() for L in layers]
        ax.plot(layers, mean_i, marker="o", color="tab:blue", label=f"|Δπ_{pair[0]}|")
        ax.plot(layers, mean_j, marker="s", color="tab:orange", label=f"|Δπ_{pair[1]}|")
        for label, tau in taus.items():
            ax.axhline(tau, ls="--", color=colours[label], lw=1.0,
                       label=f"τ_{label}={tau:.3f}")
        ax.set_title(f"{pair[0]} + {pair[1]}")
        ax.set_xlabel("layer L")
        ax.set_ylabel("mean |π^(1,1) − π^(indiv)|")
        ax.legend(fontsize=8)

    fig.suptitle("τ recipe comparison on eq (2) integrand", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# === main ==================================================================

def main() -> None:
    summary = load_summary()
    cfg = summary["config"]
    pairs = [tuple(p["pair"]) for p in summary["pairs"]]
    pair_blobs = {p: load_pair_projections(p) for p in pairs}
    integrands = integrand_per_pair(pair_blobs)

    print(f"original tau (raw layer-to-layer recipe) = {cfg['tau']:.4f}")
    print(f"  -> all L_div saturated at L_final = {cfg['trajectory_layers'][-1]}\n")

    print("=== candidate recipes ===")
    tau_r1, det_r1 = tau_R1(pair_blobs)
    print(f"R1 (pooled inter-prompt std)            : tau = {tau_r1:.4f}")

    tau_r2, det_r2 = tau_R2(pair_blobs)
    print(f"R2 (split-half null bootstrap, q={BOOTSTRAP_PERCENTILE:.0f}%)  : tau = {tau_r2:.4f}")

    tau_r3, det_r3 = tau_R3(pair_blobs, cfg["n_completions_per_prompt"])
    print(f"R3 (same-prompt inter-completion, q={BOOTSTRAP_PERCENTILE:.0f}%) : tau = {tau_r3:.4f}")

    taus = {"R1": tau_r1, "R2": tau_r2, "R3": tau_r3}

    print("\n=== L_div under each recipe (median across prompts × completions) ===")
    print(f"{'pair':<32} {'recipe':<6} {'L_div_i':>9} {'L_div_j':>9} {'frac_cross_i':>13} {'frac_cross_j':>13}")
    print("-" * 88)
    layer_final = cfg["trajectory_layers"][-1]
    by_recipe: dict[str, list[dict]] = {}
    for label, t in taus.items():
        rows = L_div_table(integrands, t)
        by_recipe[label] = rows
        for r in rows:
            print(
                f"{r['pair'][0]+'+'+r['pair'][1]:<32} {label:<6} "
                f"{r['L_div_i_median']:>9d} {r['L_div_j_median']:>9d} "
                f"{r['frac_crossed_i']:>13.2f} {r['frac_crossed_j']:>13.2f}"
            )

    out_blob = {
        "tau_factor": TAU_FACTOR,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_percentile": BOOTSTRAP_PERCENTILE,
        "seed": SEED,
        "original_tau": cfg["tau"],
        "recipes": {
            "R1": {"tau": tau_r1, "per_seed": det_r1, "L_div": by_recipe["R1"]},
            "R2": {"tau": tau_r2, "per_seed": det_r2, "L_div": by_recipe["R2"]},
            "R3": {"tau": tau_r3, "per_seed": det_r3, "L_div": by_recipe["R3"]},
        },
        "layer_final": layer_final,
    }
    RECAL_OUT.write_text(json.dumps(out_blob, indent=2))
    print(f"\nwrote {RECAL_OUT}")

    plot_taus(integrands, taus, FIG_DIR / "fig_tau_recalibration.png")
    print(f"wrote {FIG_DIR / 'fig_tau_recalibration.png'}")


if __name__ == "__main__":
    main()
