# Legacy

Frozen artefacts from Phases 0–6 of the project — the original CAA-style steering pipeline that produced the E3.1 "steered ≈ unsteered" negative result and was subsequently replaced by the Anthropic-replication pipeline now living at the repo root (`src/extraction/`, `src/inference/`, `scripts/{extraction,validation,layer_selection,trajectory,plotting}/`).

Kept for: reproducibility of the headline negative result, the methodological-diagnosis chapter of the report, and historical record. Imports inside `legacy/` are **not** maintained — scripts in here may not run against the current `src/` namespaces.

## Layout

- `caa_pipeline/scripts/` — drivers (`run_extraction.py`, `run_baseline_scoring.py`, `run_alpha_sweep.py`, `run_logprob_validation.py`, `run_layer_selection.py`, `convert_to_mwe.py`, `extract_persona_vectors.py`, `validate_persona_vectors_dual.py`, `run_composition.py`, etc.). Phase 0–6 + E8.4 schema-only `run_composition.py`.
- `caa_pipeline/src/` — modules superseded by the active pipeline:
  - `extraction.py` → replaced by `src/extraction/build_vector.py`
  - `injection.py`  → replaced by `src/inference/hf_model.py` (`steering_hook`)
  - `logprob.py`    → replaced by `src/inference/hf_logprob.py`
  - `geometry.py`, `analysis.py`, `steer_vec_loader.py` — replaced by `src/geometry/*`
- `caa_pipeline/slurm/` — SLURM wrappers for the legacy drivers.
- `caa_pipeline/data/` — early contrastive-pair datasets (`behaviors/`, `persona_artifacts/`, `persona_MWE/`, `dataset_persona/`). The **active** MWE dataset (15 traits) lives at `data/behaviors_mwe/`.
- `data_generation_root/` — root-level one-off data-generation scripts from Phase 0/2 (`data_generation.py`, `generate_corrigibility.py`, `download_persona_artifacts.py`, `fix_data.py`, etc.).
- `notebooks/` — early single-trait inspection (`anthropic_repl_evil.ipynb`, E7.3) superseded by Phases 9–11 multi-trait notebooks at `analysis/notebooks/`.
- `results/` — early result blobs: `baselines_layer17.json` (E3.1), `alpha_sweep.json` (E3.2), `evil_layer17_rescore.json`, `logprob_validation_instruct.json`, `logprob_validation_new.json`, `new_behavior_validation.json`, `persona_vectors_dual_validation.json`, persona-gen logs, plus the legacy CAA L=17 raw vectors under `vectors/`.
- `local_tests/test_extraction_injection.py` — Phase 0 smoke test for the legacy CAA pipeline.
- `notes.md` — early scratch notes.

## See also

`paper/experiments_log.md` Phases 0–6 for the full narrative; Phase 7 onwards for the replacement pipeline.
