"""
RQ2 Phase 1 pilot — assessment + richer plots from saved artifacts.

Reads results/trajectory_pilot_l17/.../pilot_summary.json and
the per-pair projections.pt files, then:
    - Prints Δ_i / Δ_j / L_div per pair (table) and the global τ.
    - Runs the L=17 analytical sanity check (roadmap §5):
          π_i^(1,0)(17) ≈ π_i^(0,0)(17) + α
          π_i^(1,1)(17) ≈ π_i^(0,0)(17) + α + α·cos(v_i, v_j)
      The (0,0) baseline is not in the pilot saved set by default — when
      missing, this script prints the *deltas* π^(1,0)(17) - π^(1,1)(17)
      and compares them to the closed-form prediction
          π^(1,1)(17) - π^(1,0)(17) = α · cos(v_i, v_j)
      which only requires the steered settings the pilot already saved.
    - Produces overlay plots:
        fig_traj_pilot_overlay_pi_i.png  — π_i vs L for all pairs, joint vs (1,0)
        fig_traj_pilot_overlay_pi_j.png  — π_j vs L for all pairs, joint vs (0,1)
        fig_traj_pilot_diff_per_pair.png — |π^(1,1) - π^(1,0)| vs L (eq 2 integrand)

Local-runnable: no model needed. Reads tensors with torch + matplotlib only.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


# === paths (mirror the pilot driver) =======================================

OUT_DIR = Path("results/trajectory_pilot_l17/Llama-3.1-8B-Instruct")
FIG_DIR = Path("analysis/figures/trajectory_pilot")
SUMMARY_PATH = OUT_DIR / "pilot_summary.json"
RECAL_PATH = OUT_DIR / "tau_recalibration.json"
RECAL_RECIPE = "R2"   # split-half null bootstrap; see recalibrate_tau_pilot_l17.py


def resolve_tau(cfg: dict) -> tuple[float, str]:
    """Prefer the recalibrated tau from tau_recalibration.json if present,
    fall back to the original recipe in pilot_summary.json otherwise."""
    if RECAL_PATH.exists():
        recal = json.loads(RECAL_PATH.read_text())
        t = recal["recipes"][RECAL_RECIPE]["tau"]
        return float(t), f"recalibrated {RECAL_RECIPE}"
    return float(cfg["tau"]), "original (raw-projection layer-to-layer)"


# === loaders ===============================================================

def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text())


def load_pair_projections(pair: tuple[str, str]) -> dict:
    """Returns
        {
            "cos": float,
            "projections": {(ai, aj): {"pi_i": {L: tensor[N]}, "pi_j": {L: tensor[N]}}}
        }
    """
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


def collect_pairs(summary: dict) -> list[tuple[str, str]]:
    return [tuple(p["pair"]) for p in summary["pairs"]]


# === assessment ============================================================

def print_table(summary: dict) -> None:
    cfg = summary["config"]
    print(
        f"\nL*={cfg['layer_star']}  alpha_unit={cfg['alpha_unit']}  "
        f"normalize_composition={cfg['normalize_composition']}  "
        f"tau={cfg['tau']:.4f}  (factor={cfg['tau_factor']})"
    )
    print(
        f"trajectory layers: L in [{cfg['trajectory_layers'][0]}, "
        f"{cfg['trajectory_layers'][-1]}]\n"
    )
    header = (
        f"{'pair':<32} {'cos':>7}  "
        f"{'Δ_i mean±std':>16}  {'Δ_j mean±std':>16}  "
        f"{'L_div_i (med)':>14}  {'L_div_j (med)':>14}"
    )
    print(header)
    print("-" * len(header))
    for ps in summary["pairs"]:
        b1, b2 = ps["pair"]
        di = ps["delta_i"]
        dj = ps["delta_j"]
        ld_i = sorted(ps["L_div_i"])
        ld_j = sorted(ps["L_div_j"])
        med_i = ld_i[len(ld_i) // 2]
        med_j = ld_j[len(ld_j) // 2]
        print(
            f"{b1+'+'+b2:<32} {ps['cos']:+7.3f}  "
            f"{di['mean']:7.3f} ± {di['std']:5.3f}  "
            f"{dj['mean']:7.3f} ± {dj['std']:5.3f}  "
            f"{med_i:>14d}  {med_j:>14d}"
        )


def sanity_check_l17(
    pair: tuple[str, str],
    pair_blob: dict,
    alpha_unit: float,
    normalize: bool,
) -> None:
    """Closed-form: under normalize=False (compose_steering_vector default off),
        delta_(1,0) = alpha · v_i_hat
        delta_(1,1) = alpha · (v_i_hat + v_j_hat)
    so injecting at the L*-th block output and reading π_i at L=L* gives
        π_i^(1,0)(17) - π_i^(0,0)(17) = alpha · ⟨v_i_hat, v_i_hat⟩ = alpha
        π_i^(1,1)(17) - π_i^(1,0)(17) = alpha · ⟨v_j_hat, v_i_hat⟩ = alpha · cos
    The pilot doesn't save the (0,0) projection, but the second equation only
    uses steered settings the pilot has, so we can verify it directly.
    """
    proj = pair_blob["projections"]
    cos_ij = pair_blob["cos"]
    L_star = 17

    pi_i_10 = proj[(1, 0)]["pi_i"][L_star]   # [N]
    pi_i_11 = proj[(1, 1)]["pi_i"][L_star]
    pi_j_01 = proj[(0, 1)]["pi_j"][L_star]
    pi_j_11 = proj[(1, 1)]["pi_j"][L_star]

    obs_di = (pi_i_11 - pi_i_10).mean().item()
    obs_dj = (pi_j_11 - pi_j_01).mean().item()

    pred = alpha_unit * cos_ij if not normalize else None

    print(f"\n  sanity check at L={L_star} for {pair[0]}+{pair[1]} (cos={cos_ij:+.4f}):")
    if pred is not None:
        print(
            f"    obs  π_i^(1,1) - π_i^(1,0) = {obs_di:+.4f}   "
            f"pred α·cos = {pred:+.4f}   diff = {obs_di - pred:+.4f}"
        )
        print(
            f"    obs  π_j^(1,1) - π_j^(0,1) = {obs_dj:+.4f}   "
            f"pred α·cos = {pred:+.4f}   diff = {obs_dj - pred:+.4f}"
        )
    else:
        print(
            f"    normalize=True — closed form does not factor through cos(v_i, v_j); "
            f"observed diffs: π_i={obs_di:+.4f}, π_j={obs_dj:+.4f}"
        )


# === plots =================================================================

def _per_layer_summary(pi: dict[int, torch.Tensor]) -> tuple[list[int], list[float], list[float]]:
    layers = sorted(pi.keys())
    means = [pi[L].mean().item() for L in layers]
    stds = [pi[L].std(unbiased=False).item() for L in layers]
    return layers, means, stds


def overlay_pi_axis(
    pairs: list[tuple[str, str]],
    pair_blobs: dict[tuple[str, str], dict],
    indiv_setting: tuple[int, int],
    direction_key: str,         # "pi_i" or "pi_j"
    title: str,
    out_path: Path,
    layer_star: int,
) -> None:
    """One subplot per pair, joint (1,1) trajectory overlaid on individual."""
    import matplotlib.pyplot as plt

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        proj = pair_blobs[pair]["projections"]
        cos_ij = pair_blobs[pair]["cos"]

        L_indiv, m_indiv, s_indiv = _per_layer_summary(proj[indiv_setting][direction_key])
        L_joint, m_joint, s_joint = _per_layer_summary(proj[(1, 1)][direction_key])

        ax.plot(L_indiv, m_indiv, marker="o", label=f"individual {indiv_setting}")
        ax.fill_between(
            L_indiv,
            [m - s for m, s in zip(m_indiv, s_indiv)],
            [m + s for m, s in zip(m_indiv, s_indiv)],
            alpha=0.2,
        )
        ax.plot(L_joint, m_joint, marker="s", label="joint (1,1)")
        ax.fill_between(
            L_joint,
            [m - s for m, s in zip(m_joint, s_joint)],
            [m + s for m, s in zip(m_joint, s_joint)],
            alpha=0.2,
        )
        ax.axvline(layer_star, ls="--", color="grey", lw=0.8)
        ax.set_title(f"{pair[0]} + {pair[1]}\ncos={cos_ij:+.3f}")
        ax.set_xlabel("layer L")
        ax.set_ylabel("projection")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def diff_per_pair_plot(
    pairs: list[tuple[str, str]],
    pair_blobs: dict[tuple[str, str], dict],
    tau: float,
    out_path: Path,
    layer_star: int,
) -> None:
    """|π^(1,1) - π^(1,0)| and |π^(1,1) - π^(0,1)| per pair across L.
    Eq (2) integrand. Horizontal τ line marks the L_div threshold."""
    import matplotlib.pyplot as plt

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        proj = pair_blobs[pair]["projections"]
        cos_ij = pair_blobs[pair]["cos"]

        layers = sorted(proj[(1, 0)]["pi_i"].keys())
        diff_i = [
            (proj[(1, 1)]["pi_i"][L] - proj[(1, 0)]["pi_i"][L]).abs().mean().item()
            for L in layers
        ]
        diff_j = [
            (proj[(1, 1)]["pi_j"][L] - proj[(0, 1)]["pi_j"][L]).abs().mean().item()
            for L in layers
        ]
        ax.plot(layers, diff_i, marker="o", label=f"|Δπ_{pair[0]}|")
        ax.plot(layers, diff_j, marker="s", label=f"|Δπ_{pair[1]}|")
        ax.axhline(tau, ls=":", color="red", lw=0.8, label=f"τ={tau:.3f}")
        ax.axvline(layer_star, ls="--", color="grey", lw=0.8)
        ax.set_title(f"{pair[0]} + {pair[1]}\ncos={cos_ij:+.3f}")
        ax.set_xlabel("layer L")
        ax.set_ylabel("|π^(1,1) - π^(indiv)|  (mean over prompts)")
        ax.legend(fontsize=8)

    fig.suptitle("Eq (2) integrand by layer — first crossing of τ defines L_div (eq 3)",
                 fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# === verdict ===============================================================

def verdict(summary: dict) -> None:
    """Roadmap §4 reading rules:
        antipodal pair (cos << 0)          : Δ should be large, L_div early.
        near-orthogonal pair (cos ≈ 0)     : Δ should be small, L_div saturated.
        moderate (cos ~ 0.4)               : intermediate.
    No fixed pass/fail thresholds in the roadmap — print the contrast and let
    the writer judge.
    """
    rows = []
    for ps in summary["pairs"]:
        rows.append((
            ps["pair"],
            ps["cos"],
            ps["delta_i"]["mean"],
            ps["delta_j"]["mean"],
        ))
    rows.sort(key=lambda r: r[1])  # by cos ascending (antipodal first)

    print("\nverdict (sorted by cos):")
    for pair, c, di, dj in rows:
        avg_delta = (di + dj) / 2
        regime = "antipodal" if c < -0.15 else ("orthogonal" if abs(c) < 0.15 else "moderate")
        print(
            f"  {pair[0]}+{pair[1]:<18s}  cos={c:+.3f}  "
            f"avg Δ={avg_delta:.3f}  regime={regime}"
        )
    print(
        "\n  expected pattern: avg Δ should rank antipodal > moderate > orthogonal.\n"
        "  if it does, calibrated τ is pre-registered → proceed to full sweep.\n"
        "  if Δ values look like noise across the board, proceed to full sweep\n"
        "  anyway and commit to a clean negative result (roadmap §4 closing)."
    )


# === main flow =============================================================

def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"pilot summary not found at {SUMMARY_PATH} — run "
            "scripts/trajectory/run_trajectory_pilot_l17.py first"
        )

    summary = load_summary()
    cfg = summary["config"]
    pairs = collect_pairs(summary)

    tau, tau_origin = resolve_tau(cfg)
    print(f"\nusing tau = {tau:.4f} ({tau_origin})")

    pair_blobs = {p: load_pair_projections(p) for p in pairs}

    print_table(summary)

    print("\n=== L=17 analytical sanity check ===")
    for p in pairs:
        sanity_check_l17(p, pair_blobs[p], cfg["alpha_unit"], cfg["normalize_composition"])

    print("\n=== overlay plots ===")
    overlay_pi_axis(
        pairs, pair_blobs,
        indiv_setting=(1, 0),
        direction_key="pi_i",
        title="π_i across pairs — individual (1,0) vs joint (1,1)",
        out_path=FIG_DIR / "fig_traj_pilot_overlay_pi_i.png",
        layer_star=cfg["layer_star"],
    )
    print(f"  wrote {FIG_DIR / 'fig_traj_pilot_overlay_pi_i.png'}")

    overlay_pi_axis(
        pairs, pair_blobs,
        indiv_setting=(0, 1),
        direction_key="pi_j",
        title="π_j across pairs — individual (0,1) vs joint (1,1)",
        out_path=FIG_DIR / "fig_traj_pilot_overlay_pi_j.png",
        layer_star=cfg["layer_star"],
    )
    print(f"  wrote {FIG_DIR / 'fig_traj_pilot_overlay_pi_j.png'}")

    diff_per_pair_plot(
        pairs, pair_blobs,
        tau=tau,
        out_path=FIG_DIR / "fig_traj_pilot_diff_per_pair.png",
        layer_star=cfg["layer_star"],
    )
    print(f"  wrote {FIG_DIR / 'fig_traj_pilot_diff_per_pair.png'}")

    verdict(summary)


if __name__ == "__main__":
    main()
