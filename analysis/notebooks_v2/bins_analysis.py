"""
bins_analysis.py

Runner for the composition-geometry statistical analysis.
Just run this file directly — no terminal arguments needed.

It reads the consolidated (coh≥30) frame, runs the full battery on the
**norm** (`normTrue`) and **per-axis** (`per_axis`) subsets, and writes ONE
combined report in which each step's two result tables sit side by side:

    analysis/results/RQ1/norm_vs_peraxis_bins_analysis/norm_vs_peraxis_bins_analysis.{md,json}

The `## Conclusions` section of the .md is preserved across re-runs.
"""

from bins_analysis_scripts import run_consolidated_analysis

# --------------------------------------------------------------------------- #
# Parameters — edit these, then run the file.
# --------------------------------------------------------------------------- #
N_BOOT = 2000           # bootstrap iterations
N_PERM = 10000          # permutation iterations


if __name__ == "__main__":
    run_consolidated_analysis(n_boot=N_BOOT, n_perm=N_PERM)
