"""
RQ2 Phase 1 pilot — projection-trajectory pipeline on three pairs at L*=17.

Per RQ2 roadmap (compose-or-collide, 2026-05-05):
    - Pair set: (formality, impolite) antipodal cross-cluster, cos(L17)=-0.232;
                (apathetic, power_seeking) near-orthogonal, cos(L17)=+0.006;
                (evil, sycophantic) moderate cosine, cos(L17)=+0.418.
    - Settings: (1,0), (0,1), (1,1).
    - Operating point: L*=17, alpha_unit=4 on unit-normalised response_avg_diff[17]
      (E10.4). Composition formula = alpha_unit * (w_i v_i_hat + w_j v_j_hat),
      i.e. compose_steering_vector(..., normalize=False) — matches the roadmap
      sanity check at L=17 (pi^(1,1)(17) = baseline + alpha*(1 + cos)).
    - Per (pair, setting): generate N_COMPLETIONS_PER_PROMPT completions on
      N_PROMPTS_PER_TRAIT * 2 evaluation prompts (taken from each behaviour's
      held-out trait_data_eval set). Then teacher-forced re-pass on each
      (prompt, completion) with the same delta to cache residual-stream
      activations across L >= L*=17 (response-token-averaged, matches upstream
      extraction convention).
    - Project onto v_i and v_j (eq 1). Compute Delta_i, Delta_j (eq 2) over
      settings (1,1) vs (1,0) [for v_i] and (1,1) vs (0,1) [for v_j].
      Calibrate tau from the individual-steering trajectories (1.5x max
      layer-to-layer noise, eq 3 reg).
    - Outputs: completions JSONL, projection tensors (.pt), summary JSON,
      per-pair trajectory PNG.

Decision flags (pre-registered as constants — do not flip mid-run):
    NORMALIZE_COMPOSITION = False  (roadmap math; flip to True for the
        fixed-magnitude alternate from the compose_steering_vector docstring).
    TAU_FACTOR = 1.5               (roadmap reg).

No argparse, no classes — functions only. Idempotent per (pair, setting):
checks for existing completions.jsonl and skips regeneration; trajectory
projections are recomputed if completions exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.extraction.generation import generate_batch
from src.inference.hf_model import load_hf_model
from src.extraction.trait_data import load_trait
from src.joint_analysis.joint_injection import (
    PERSONA_VECTOR_DIR,
    calibrate_tau,
    compose_steering_vector,
    layer_of_divergence,
    load_unit_vector,
    project_trajectory,
    stack_trajectory,
    traj_divergence,
    trajectory_response_avg,
)


# === pre-registered configuration =========================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

LAYER_STAR = 17
ALPHA_UNIT = 4.0
NORMALIZE_COMPOSITION = False  # see module docstring
TAU_FACTOR = 1.5

PILOT_PAIRS: list[tuple[str, str]] = [
    ("formality", "impolite"),       # antipodal cross-cluster
    ("apathetic", "power_seeking"),  # near-orthogonal
    ("evil", "sycophantic"),         # moderate cosine
]

SETTINGS: list[tuple[int, int]] = [(1, 0), (0, 1), (1, 1)]

N_PROMPTS_PER_TRAIT = 5         # 10 prompts per pair (5 from each behaviour)
N_COMPLETIONS_PER_PROMPT = 3
MAX_NEW_TOKENS = 200
TEMPERATURE = 1.0
BATCH_SIZE = 4

OUT_DIR = Path("results/trajectory_pilot_l17/Llama-3.1-8B-Instruct")
FIG_DIR = Path("analysis/figures")
LOGS_DIR = Path("logs")


# === helpers ===============================================================

def pair_dir(pair: tuple[str, str]) -> Path:
    return OUT_DIR / f"{pair[0]}__{pair[1]}"


def setting_dir(pair: tuple[str, str], s: tuple[int, int]) -> Path:
    return pair_dir(pair) / f"setting_{s[0]}_{s[1]}"


def collect_pair_prompts(b1: str, b2: str, n_per_trait: int) -> list[str]:
    """Take the first n_per_trait questions from each behaviour's eval JSON,
    de-dupe while preserving order. Both members of a pair contribute prompts
    so trajectories see prompts where each behaviour has natural headroom."""
    qs1 = load_trait(b1, "eval").questions[:n_per_trait]
    qs2 = load_trait(b2, "eval").questions[:n_per_trait]
    seen, out = set(), []
    for q in list(qs1) + list(qs2):
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def build_delta(
    setting: tuple[int, int],
    v_i: torch.Tensor,
    v_j: torch.Tensor,
) -> torch.Tensor | None:
    """Compose the residual perturbation for a setting. setting=(0,0) -> None
    (unsteered baseline; included only if SETTINGS is extended)."""
    a, b = setting
    if a == 0 and b == 0:
        return None
    return compose_steering_vector(
        [(v_i, float(a)), (v_j, float(b))],
        alpha=ALPHA_UNIT,
        normalize=NORMALIZE_COMPOSITION,
    )


def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    delta: torch.Tensor | None,
    n_completions: int,
) -> list[list[str]]:
    """For each prompt, return n_completions sampled completions under the
    given delta. Replication is via prompt list duplication so generate_batch
    can group sampling calls. Returns list-of-lists: [n_prompts][n_completions]."""
    flat_convs = []
    for p in prompts:
        for _ in range(n_completions):
            flat_convs.append([{"role": "user", "content": p}])

    if delta is not None:
        steering = (delta, LAYER_STAR - 1, 1.0, "response")
    else:
        steering = None

    _, answers = generate_batch(
        model,
        tokenizer,
        flat_convs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        batch_size=BATCH_SIZE,
        steering=steering,
    )

    grouped = []
    for i in range(len(prompts)):
        grouped.append(answers[i * n_completions : (i + 1) * n_completions])
    return grouped


def save_completions_jsonl(
    path: Path,
    pair: tuple[str, str],
    setting: tuple[int, int],
    prompts: list[str],
    completions: list[list[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, (p, comps) in enumerate(zip(prompts, completions)):
            for j, c in enumerate(comps):
                f.write(json.dumps({
                    "pair": list(pair),
                    "setting": list(setting),
                    "prompt_idx": i,
                    "completion_idx": j,
                    "prompt": p,
                    "completion": c,
                }) + "\n")


def load_completions_jsonl(path: Path) -> tuple[list[str], list[list[str]]]:
    """Inverse of save_completions_jsonl. Returns (prompts, completions) where
    completions[i] is a list of N_COMPLETIONS_PER_PROMPT strings."""
    rows = [json.loads(line) for line in open(path)]
    by_idx: dict[int, dict[int, str]] = {}
    prompts: dict[int, str] = {}
    for r in rows:
        prompts[r["prompt_idx"]] = r["prompt"]
        by_idx.setdefault(r["prompt_idx"], {})[r["completion_idx"]] = r["completion"]
    n = max(prompts.keys()) + 1
    out_prompts = [prompts[i] for i in range(n)]
    out_comps = [
        [by_idx[i][j] for j in sorted(by_idx[i].keys())] for i in range(n)
    ]
    return out_prompts, out_comps


def extract_trajectory_for_setting(
    model,
    tokenizer,
    prompts: list[str],
    completions: list[list[str]],
    delta: torch.Tensor | None,
    layers_above: list[int],
) -> list[dict[int, torch.Tensor]]:
    """One trajectory per (prompt, completion). Flat list ordered as
    (prompt_0, comp_0), (prompt_0, comp_1), ..."""
    trajs = []
    for p, comps in zip(prompts, completions):
        for c in comps:
            avg = trajectory_response_avg(
                model,
                tokenizer,
                p,
                c,
                layers_above=layers_above,
                delta_at_lstar=delta,
                layer_star=LAYER_STAR,
            )
            trajs.append(avg)
    return trajs


def project_pair(
    trajs_per_setting: dict[tuple[int, int], list[dict[int, torch.Tensor]]],
    v_i: torch.Tensor,
    v_j: torch.Tensor,
) -> dict[tuple[int, int], dict[str, dict[int, torch.Tensor]]]:
    """For each setting -> {"pi_i": dict[L]->[N], "pi_j": dict[L]->[N]}."""
    out: dict[tuple[int, int], dict[str, dict[int, torch.Tensor]]] = {}
    for s, trajs in trajs_per_setting.items():
        stacked = stack_trajectory(trajs)
        out[s] = {
            "pi_i": project_trajectory(stacked, v_i),
            "pi_j": project_trajectory(stacked, v_j),
        }
    return out


def per_prompt_summary(pi: dict[int, torch.Tensor]) -> dict[str, list[float]]:
    """Mean and std across prompts for each layer — used by the plot."""
    layers = sorted(pi.keys())
    means = [pi[L].mean().item() for L in layers]
    stds = [pi[L].std(unbiased=False).item() for L in layers]
    return {"layers": layers, "mean": means, "std": stds}


def plot_pair(
    pair: tuple[str, str],
    proj: dict[tuple[int, int], dict[str, dict[int, torch.Tensor]]],
    cos_ij: float,
    out_path: Path,
) -> None:
    """Two-panel figure: pi_i under (1,0) & (1,1); pi_j under (0,1) & (1,1)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    b1, b2 = pair

    def _draw(ax, summary, label):
        layers = summary["layers"]
        mean = summary["mean"]
        std = summary["std"]
        ax.plot(layers, mean, marker="o", label=label)
        ax.fill_between(
            layers,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            alpha=0.2,
        )

    s_i_indiv = (1, 0)
    s_j_indiv = (0, 1)
    s_joint = (1, 1)

    _draw(axes[0], per_prompt_summary(proj[s_i_indiv]["pi_i"]), f"(1,0) on v_{b1}")
    _draw(axes[0], per_prompt_summary(proj[s_joint]["pi_i"]),   f"(1,1) on v_{b1}")
    axes[0].set_title(f"pi_{b1}  (steering direction = v_{b1})")
    axes[0].set_xlabel("layer L")
    axes[0].set_ylabel("projection")
    axes[0].axvline(LAYER_STAR, ls="--", color="grey", lw=0.8)
    axes[0].legend()

    _draw(axes[1], per_prompt_summary(proj[s_j_indiv]["pi_j"]), f"(0,1) on v_{b2}")
    _draw(axes[1], per_prompt_summary(proj[s_joint]["pi_j"]),   f"(1,1) on v_{b2}")
    axes[1].set_title(f"pi_{b2}  (steering direction = v_{b2})")
    axes[1].set_xlabel("layer L")
    axes[1].axvline(LAYER_STAR, ls="--", color="grey", lw=0.8)
    axes[1].legend()

    fig.suptitle(
        f"RQ2 pilot trajectories — {b1} + {b2}  "
        f"(L*={LAYER_STAR}, alpha_unit={ALPHA_UNIT}, cos={cos_ij:+.3f}, "
        f"normalize={NORMALIZE_COMPOSITION})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def pair_summary(
    pair: tuple[str, str],
    proj: dict[tuple[int, int], dict[str, dict[int, torch.Tensor]]],
    cos_ij: float,
    tau: float,
) -> dict:
    """Compute Delta_i, Delta_j, L_div per behaviour. Tau passed in (calibrated
    globally across pairs for cross-pair comparability)."""
    s_i = (1, 0)
    s_j = (0, 1)
    s_joint = (1, 1)

    delta_i = traj_divergence(proj[s_joint]["pi_i"], proj[s_i]["pi_i"])
    delta_j = traj_divergence(proj[s_joint]["pi_j"], proj[s_j]["pi_j"])
    L_div_i = layer_of_divergence(proj[s_joint]["pi_i"], proj[s_i]["pi_i"], tau=tau)
    L_div_j = layer_of_divergence(proj[s_joint]["pi_j"], proj[s_j]["pi_j"], tau=tau)

    return {
        "pair": list(pair),
        "cos": round(cos_ij, 4),
        "n_samples": int(delta_i.numel()),
        "delta_i": {
            "mean": round(delta_i.mean().item(), 4),
            "std": round(delta_i.std(unbiased=False).item(), 4),
            "per_sample": [round(x, 4) for x in delta_i.tolist()],
        },
        "delta_j": {
            "mean": round(delta_j.mean().item(), 4),
            "std": round(delta_j.std(unbiased=False).item(), 4),
            "per_sample": [round(x, 4) for x in delta_j.tolist()],
        },
        "L_div_i": L_div_i.tolist(),
        "L_div_j": L_div_j.tolist(),
    }


# === main flow =============================================================

def run_pair(
    model,
    tokenizer,
    pair: tuple[str, str],
    layers_above: list[int],
) -> dict:
    """Generate completions for each setting (cached), extract trajectories,
    project, save. Returns {"projections": ..., "cos": ...} for downstream
    summary + tau calibration."""
    b1, b2 = pair
    print(f"\n=== {b1} + {b2} ===")

    v_i = load_unit_vector(b1, layer=LAYER_STAR)
    v_j = load_unit_vector(b2, layer=LAYER_STAR)
    cos_ij = (v_i @ v_j).item()
    print(f"  cos(v_{b1}, v_{b2}) at L={LAYER_STAR}: {cos_ij:+.4f}")

    prompts = collect_pair_prompts(b1, b2, N_PROMPTS_PER_TRAIT)
    print(f"  using {len(prompts)} eval prompts ({N_COMPLETIONS_PER_PROMPT} completions each)")

    trajs_per_setting: dict[tuple[int, int], list[dict[int, torch.Tensor]]] = {}

    for s in SETTINGS:
        d = setting_dir(pair, s)
        d.mkdir(parents=True, exist_ok=True)
        comp_path = d / "completions.jsonl"

        delta = build_delta(s, v_i, v_j)

        if comp_path.exists():
            print(f"  setting {s} completions cached at {comp_path}, skipping generation")
            prompts_loaded, completions = load_completions_jsonl(comp_path)
            assert prompts_loaded == prompts, (
                "cached prompts do not match the configured eval set — delete "
                f"{comp_path} or roll back N_PROMPTS_PER_TRAIT"
            )
        else:
            print(f"  setting {s} generating ({len(prompts)} prompts x "
                  f"{N_COMPLETIONS_PER_PROMPT} completions)...", flush=True)
            completions = generate_completions(
                model, tokenizer, prompts, delta, N_COMPLETIONS_PER_PROMPT
            )
            save_completions_jsonl(comp_path, pair, s, prompts, completions)

        print(f"  setting {s} extracting trajectories ...", flush=True)
        trajs = extract_trajectory_for_setting(
            model, tokenizer, prompts, completions, delta, layers_above
        )
        trajs_per_setting[s] = trajs

    proj = project_pair(trajs_per_setting, v_i, v_j)

    # Persist projection tensors for re-analysis without re-running the model.
    proj_path = pair_dir(pair) / "projections.pt"
    serial = {
        f"{s[0]}_{s[1]}": {
            key: {str(L): proj[s][key][L].cpu() for L in proj[s][key]}
            for key in ("pi_i", "pi_j")
        }
        for s in proj
    }
    torch.save({"projections": serial, "cos": cos_ij}, proj_path)
    print(f"  saved projections -> {proj_path}")

    return {"projections": proj, "cos": cos_ij}


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-flight: confirm every pilot vector exists before touching the model.
    for b1, b2 in PILOT_PAIRS:
        for trait in (b1, b2):
            p = PERSONA_VECTOR_DIR / f"{trait}_response_avg_diff.pt"
            if not p.exists():
                raise FileNotFoundError(
                    f"missing persona vector for {trait}: {p}"
                )

    print(f"loading {MODEL_NAME} ...", flush=True)
    model, tokenizer = load_hf_model(MODEL_NAME)
    n_layers = model.config.num_hidden_layers
    layers_above = list(range(LAYER_STAR, n_layers + 1))
    print(f"hidden_size={model.config.hidden_size}, n_layers={n_layers}, "
          f"trajectory layers L in [{layers_above[0]}, {layers_above[-1]}]")

    pair_results: dict[tuple[str, str], dict] = {}
    for pair in PILOT_PAIRS:
        pair_results[pair] = run_pair(model, tokenizer, pair, layers_above)

    # tau calibration: 1.5 * max layer-to-layer step across all individual-
    # steering trajectories (settings (1,0) projected on v_i, (0,1) on v_j),
    # pooled across pairs. Pre-registered for the full sweep.
    indiv_trajectories: list[dict[int, torch.Tensor]] = []
    for pair, info in pair_results.items():
        proj = info["projections"]
        for L_traj in (proj[(1, 0)]["pi_i"], proj[(0, 1)]["pi_j"]):
            # split [N] back into N single-prompt traj dicts so calibrate_tau
            # picks up the per-prompt layer-to-layer noise rather than the
            # mean trace.
            n = next(iter(L_traj.values())).numel()
            for k in range(n):
                indiv_trajectories.append({L: L_traj[L][k] for L in L_traj})
    tau = calibrate_tau(indiv_trajectories, factor=TAU_FACTOR)
    print(f"\ntau = {TAU_FACTOR} * max layer-to-layer noise = {tau:.4f}")

    summary = {
        "config": {
            "model": MODEL_NAME,
            "layer_star": LAYER_STAR,
            "alpha_unit": ALPHA_UNIT,
            "normalize_composition": NORMALIZE_COMPOSITION,
            "tau_factor": TAU_FACTOR,
            "tau": round(tau, 6),
            "settings": [list(s) for s in SETTINGS],
            "n_prompts_per_trait": N_PROMPTS_PER_TRAIT,
            "n_completions_per_prompt": N_COMPLETIONS_PER_PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "trajectory_layers": layers_above,
        },
        "pairs": [],
    }

    for pair, info in pair_results.items():
        ps = pair_summary(pair, info["projections"], info["cos"], tau)
        summary["pairs"].append(ps)

        fig_path = FIG_DIR / f"fig_traj_pilot_{pair[0]}__{pair[1]}.png"
        plot_pair(pair, info["projections"], info["cos"], fig_path)
        print(f"  saved figure -> {fig_path}")

        print(
            f"  {pair}: Delta_{pair[0]} mean={ps['delta_i']['mean']:.3f} "
            f"std={ps['delta_i']['std']:.3f}  |  "
            f"Delta_{pair[1]} mean={ps['delta_j']['mean']:.3f} "
            f"std={ps['delta_j']['std']:.3f}"
        )

    summary_path = OUT_DIR / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
