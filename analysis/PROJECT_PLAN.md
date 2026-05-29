# Geometry of Behavioral Control — Consolidated Project Plan

**Model:** Llama-3.1-8B-Instruct only, TransformerLens, layer L\*=17,
α=4.0, unnormalized injection per proposal Eq. 3.
(Base-vs-instruct comparison **dropped** — see Phase 4.)

This document merges two threads:
1. The **main project skeleton** (Phases 0–5, steps S1–S10).
2. The **four-notebook reconciliation** (`composition_anal`, `phase12_vs_phase125_audit`,
   `signed_cosine_predicts_suppression`, `per_axis_deep_dive`).

Items contributed by the notebook-merge thread are marked **⊕**.

---

## Naming conventions (settle these first — they collide otherwise)

- **Datasets** = injection schemes. Renamed to avoid clashing with S6's "Model A–D":

  | name | file | injection | holds constant | n | additives |
  |------|------|-----------|----------------|---|-----------|
  | **`raw`** (primary, pre-registered) | `results/composition_scoring_l17_summary.json` | `δ = α(v̂_a+v̂_b)` | unit-vector coeff | 36 | 3 |
  | **`norm`** | `results/composition_scoring_l17_v2_summary.json` | `δ = α(v̂_a+v̂_b)/‖·‖` | total `‖δ‖ = α` | 28 | 7 |
  | **`peraxis`** | `results/composition_scoring_l17_v3_summary.json` | per-axis push = α | per-axis push | 28 | 5 |

  Mechanical consequences: `raw` ⇒ `‖δ‖ = α√(2(1+cos))` (push grows with cos);
  `norm` ⇒ per-axis push `= α√((1+cos)/2)` (compresses for antipodal);
  `peraxis` ⇒ `‖δ‖ = α√(2/(1+cos))` (blows up for antipodal → coherence collapse).

- **Predictors** (S6 "Models"): A = |cos|, B = S (semantic sim), C = |cos|+S, D = signed cos+S.
- **Response metrics** — keep distinct, do not blur:
  - `Q` = composition-quality ratio (proposal Eq. 4), the skeleton's primary target.
  - `mean_joint_abs` = ½(|δ_a_joint| + |δ_b_joint|), Riccardo's magnitude target.
  - `supp_mean` = ratio-based suppression; **demoted** (unstable at small |δ_single|).

- **Decision on file:** `raw` is **primary and pre-registered**. `norm`/`peraxis` are
  alternative-injection datasets (robustness / extension), **not** replacements.

---

## Phase 0 — Setup ✓ DONE
9 traits / 12 behavior vectors via CAA mean-difference, saved as `.pt`.

## Phase 1 — Geometry S1–S3 ✓ DONE
- S1 Load 12 vectors → [12×4096], row-normalized.
- S2 Gram matrix G = VVᵀ, clustered heatmap. Vectors near-orthogonal (mean |cos| ≈ 0.16).
- S3 Stratify pairs by |G_ij|. High stratum nearly empty → near-orthogonality reframed as a
  substantive finding. Antisocial-cluster confound identified.
- **⊕** Geometric backdrop for why categorical composition regimes are sparse — cross-reference
  forward to Phase 3.

## Phase 2 — Composition data S4–S5 ✓ DONE
- S4 Joint steering sweep, 6 coefficient settings, judge scoring, composition score → `raw` n=36.
- S5 Regime classification (`raw`): mixed 19, emergent 6, dominant 5, additive 3, suppressive 3.
- **⊕ Dataset inventory:** three composition datasets exist on disk (`raw`/`norm`/`peraxis`,
  see table above). Make the three-injection-scheme structure explicit here so Phase 3 can use it.

## Phase 3 — RQ1 regression analysis S6–S8 (IN PROGRESS) — *most merge content lands here*

**Loose ends to close first:**
- ✓ Normalization question — RESOLVED: unnormalized (`raw`) is pre-registered, n=36 primary.
- ☐ **Verify Q wiring:** confirm `delta_*` are Eq. 4 expression levels; additive → Q≈1,
  suppressive → Q≈0. (`composition_anal.ipynb` cell 43–45 already does this sanity check.)
- ☐ **⊕ Fix the binning bug:** cell 7 bins `|cos|` into `[0,0.15,0.30,0.40]`; pairs with
  `|cos|>0.40` → NaN. Pick one bin scheme and use it consistently.
- ☐ **⊕ Fix prose/code mismatch:** `composition_anal.ipynb` markdown describes `norm`
  ("28 pairs / 7 additive") but the code loads `raw` (n=36); `rq1_summary.csv` confirms n=36.
- ☐ Semantic matrix S ✓ already computed (range ≈ 0.19–0.47).

### S6 — Binary logistic (SECONDARY, underpowered, 3 positives)
- Models A (|cos|), B (S), C (|cos|+S), D (signed cos+S).
- In-sample AUC + bootstrap CI (refit per resample) + permutation p. LOO computed but
  labeled uninterpretable. Report with explicit **n_positive=3 caveat**. Optionally Firth
  penalized logistic for the rare-events structure.
- **⊕** Run the identical battery on **all three datasets** (`raw`/`norm`/`peraxis`) side by
  side (norm has 7 positives, peraxis 5) — presents dataset-dependence cleanly.
- **⊕ Dose/coherence diagnostic** (`phase12_vs_phase125_audit.ipynb`) attaches here as the
  *explanation* of the n_pos=3 caveat: on `raw`, `coh_steered` alone out-predicts |cos|
  (LOO 0.72 vs 0.61); the `cos→‖δ‖→coh→regime` mediator chain (first link r=0.995) breaks
  under `norm`; the 3 positives are all `power_seeking` degenerate (single-δ≈0). This is *why*
  the in-sample AUC 0.83 does not generalize.
- **⊕** Demote the multinomial regime models (old Exp4/Exp7) to appendix — 4 classes × n too sparse.

### S7 — Continuous Spearman (PRIMARY RQ1 TEST)
- Spearman(|cos|, Q), Spearman(signed cos, Q) over all 36 `raw` points.
- Partial Spearman controlling for S (residualize both on S); report raw vs partial ρ + % attenuation.
- Spearman(S, Q); Pearson r(|cos|, S) for the confound check (recompute on chosen S).
- **⊕ Alternative-response-metric arm** (where the positive result lives):
  `signed_cosine_predicts_suppression.ipynb` finds **r(signed cos, mean_joint_abs) = +0.65,
  p ≈ 5×10⁻⁴ on `norm`**, with full robustness (LOO-pair, Spearman/Kendall, Bonferroni over 6
  predictors, ratio-reliability filter, cross-dataset replication r≈0.92 on joint deltas).
  Framing: the RQ1 signal depends jointly on (i) signed vs |cos|, (ii) Q vs mean_joint_abs,
  (iii) injection scheme. Collapsing sign or discretizing the response kills it — which also
  explains the S6 null.
- **⊕** Per-trait heterogeneity sub-result (apathetic/hallucinating dominance; ANOVA F=4.02,
  p=0.0016 on `norm`; no single trait survives Bonferroni — report as trend).

### S8 — Higher-order subspace predictor (NEW — not in any notebook)
- Project vectors onto top-k PCs (k=5 primary; 3, 10 robustness), compute subspace-overlap
  predictor, compare vs raw cosine via likelihood-ratio test.
- Frame as "does *any* geometric feature predict Q," not a rescue of a null cosine result.
- **⊕** Companion: per_axis Probe-1 collinearity (cos may proxy single-trait magnitude).

## Phase 4 — ~~Base-vs-instruct geometry S9–S10~~ DROPPED
**Decision (2026-05-29): instruct-only. No base-vs-instruct comparison.**
- ~~S9 Gram comparison (base vs instruct)~~ — dropped.
- ~~S10 Category cluster permutation test across both models~~ — dropped as a base-vs-instruct test.
  *Optional salvage:* the within- vs between-category cosine permutation test could survive as an
  **instruct-only descriptive analysis** (folded into Phase 1 geometry, connects to the
  antisocial-cluster confound and the Assistant Axis shared-axis frame). Keep only if wanted —
  not currently planned.

## Phase 5 — Robustness & writeup — ⊕ absorbs the per_axis probes
- Re-run S6/S7 at L\*±3.
- Re-run with judge scores perturbed ±5; secondary-judge re-score on 20% subsample.
- For S8/higher-order: connect to Assistant Axis if shared-subspace predictor wins.
- **⊕ From `per_axis_deep_dive.ipynb`:**
  - **Geometry-vs-injection-design** (Probe 1): partial r(cos | mean_single_abs) on the response
    is **+0.45 on `norm` but +0.03 on `peraxis`** — under per-axis injection, cos predicts joint
    behaviour only through correlated single-trait magnitude. Honest limit on any causal claim.
  - **Coherence confound / row-level filtering** (Probes 8, 10): `mean_joint_abs` strengthens
    under stricter coh filtering; `supp_mean` weakens to n.s. → confirms `mean_joint_abs` as the
    defensible target.
  - **Synergy = judge artifact** (Probes 5+7): "emergent" pairs are gibberish (coh 5–25) rated
    ~99 on both traits → motivates a **judge re-audit on low-coherence outputs** (fold into the
    judge-perturbation robustness).

---

## Strategic read

- **RQ1 (cosine predicts composition) is heading toward a null** on `raw`. The pre-registered
  run gives an in-sample cosine effect that rests on 3 same-trait points and fails LOO. S7's
  continuous Spearman is the clean adjudication — run it next.
- **Additive composition is rare (3/36)** — robust and reportable on its own.
- **Base-vs-instruct (S9/S10) is dropped**, so it can no longer serve as the fallback backbone.
  With it gone, the paper now rests on the instruct-only composition-geometry story:
  (i) near-orthogonality of behavior vectors (Phase 1); (ii) rare additivity (3/36);
  (iii) the three-injection-scheme comparison + dose-confound result (audit); (iv) the **signed-cos
  → mean_joint_abs positive on `norm`** (r≈0.65); (v) per-trait heterogeneity / synergy-as-judge-
  artifact. The `norm` positive is now one of the few positive findings — see the open decision below.

### Open decision — how to position the `norm` positive result
The `norm` result (signed cos → mean_joint_abs, r=0.65) is **compatible** with a `raw` null —
different predictor, target, and dataset — but the writeup must position it deliberately.
**With S9/S10 dropped, this decision matters more:** the `norm` positive is now one of the few
positive findings, so "scoped secondary" has no larger headline to sit behind.
- **(b) RQ1 = conditional positive** *(now the more natural default)*: lead with "geometry
  predicts composition *iff* sign is preserved and the target is a continuous magnitude," making
  the metric/scheme dependence the finding. The composition-geometry story becomes the paper.
- **(a) RQ1 = null, cleanly**: report `raw` |cos|→Q as null, use the audit to explain away the
  in-sample S6 effect, present the `norm` signed-cos result as a secondary finding. Weaker now
  that there is no base-vs-instruct backbone to headline.

---

## Where the four notebooks map

| notebook | role in the merged plan |
|----------|--------------------------|
| `composition_anal.ipynb` | Phase 3 S6 (categorical) + S7 (Q Spearman) — the RQ1 deliverable |
| `phase12_vs_phase125_audit.ipynb` | S6 dose/coherence diagnostic; Phase 2 data-quality |
| `signed_cosine_predicts_suppression.ipynb` | S7 alternative-target arm + per-trait sub-result |
| `per_axis_deep_dive.ipynb` | Phase 5 robustness (collinearity, row-level filtering, judge audit) |

**Still to write from scratch:** S8 (subspace predictor). *(S9/S10 dropped.)*

## Recommended order from here
verify Q wiring → **S6 regressions first** (binary additive-vs-non + multinomial regime, as the
categorical lead-in) → **then S7 Q correlations** (the possibly-stronger continuous result) → S8
→ robustness (per_axis probes, L\*±3, judge perturbation) → writeup.

Note: this runs the **secondary** (S6) before the **primary** (S7) for narrative reasons — fine,
but the RQ1 conclusion still hangs on **S7/Q**, not on the in-sample regression AUC. Keep the S6
caveats visible: report LOO alongside in-sample, flag `n_positive` (raw 3 / norm 7 / peraxis 5),
and treat the multinomial as exploratory (raw/peraxis = 5 sparse classes, norm = 4).
