"""
Entry point for generating all metrics and figures from saved experiment results.

Usage:
    python run_analysis.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.experiments.runner import run_all
from src.analysis.metrics import (
    compute_summary,
    H1_convergence_by_setting,
    H2_switching_by_cost,
    H3_double_q_stability,
    H4_model_free_vs_optimal,
    H5_state_management,
)
from src.visualization.plots import plot_all

pd_opts = {'display.max_columns': None, 'display.width': 120, 'display.float_format': '{:.3f}'.format}


def main():
    import pandas as pd
    for k, v in pd_opts.items():
        pd.set_option(k, v)

    print("Loading results...")
    results = run_all()
    print(f"Loaded {len(results)} results.\n")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("=" * 60)
    print("SUMMARY: Mean Final Return (algorithm × setting, horizon=10, c_switch=0)")
    print("=" * 60)
    df = compute_summary(results)
    subset = df[(df.horizon == 10) & (df.c_switch == 0.0)][
        ['algo_label', 'setting', 'mean_return', 'std_return']
    ].sort_values(['setting', 'mean_return'], ascending=[True, False])
    print(subset.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # H1: Convergence speed
    # ------------------------------------------------------------------
    print("=" * 60)
    print("H1: Convergence speed by setting (horizon=10, c_switch=0)")
    print("=" * 60)
    h1 = H1_convergence_by_setting(results)
    h1_sub = h1[(h1.horizon == 10) & (h1.c_switch == 0.0)][
        ['algo_label', 'setting', 'convergence_episode', 'final_return']
    ].sort_values(['setting', 'convergence_episode'])
    print(h1_sub.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # H2: Switching behaviour vs switch cost
    # ------------------------------------------------------------------
    print("=" * 60)
    print("H2: Switching behaviour vs switch cost (high setting, horizon=10)")
    print("=" * 60)
    h2 = H2_switching_by_cost(results)
    h2_sub = h2[(h2.setting == 'high') & (h2.horizon == 10)][
        ['algo_label', 'c_switch', 'switch_rate', 'nonreceptive_frac', 'final_return']
    ].sort_values(['algo_label', 'c_switch'])
    print(h2_sub.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # H3: Double Q stability
    # ------------------------------------------------------------------
    print("=" * 60)
    print("H3: Q-Learning vs Double Q-Learning stability (horizon=10, c_switch=0)")
    print("=" * 60)
    h3 = H3_double_q_stability(results)
    h3_sub = h3[(h3.horizon == 10) & (h3.c_switch == 0.0)][
        ['algo_label', 'setting', 'return_variance', 'mean_return']
    ].sort_values(['setting', 'algo_label'])
    print(h3_sub.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # H4: Gap to optimal
    # ------------------------------------------------------------------
    print("=" * 60)
    print("H4: Model-free gap to Value Iteration (horizon=10, c_switch=0)")
    print("=" * 60)
    h4 = H4_model_free_vs_optimal(results)
    h4_sub = h4[(h4.horizon == 10) & (h4.c_switch == 0.0)][
        ['algo_label', 'setting', 'agent_return', 'vi_return', 'gap', 'gap_pct']
    ].sort_values(['setting', 'gap'])
    print(h4_sub.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # H5: Patient state management
    # ------------------------------------------------------------------
    print("=" * 60)
    print("H5: Patient state management — receptive fraction improvement (horizon=10, c_switch=0)")
    print("=" * 60)
    h5 = H5_state_management(results)
    h5_sub = h5[(h5.horizon == 10) & (h5.c_switch == 0.0)][
        ['algo_label', 'setting', 'early_receptive', 'late_receptive', 'receptive_improvement']
    ].sort_values(['setting', 'receptive_improvement'], ascending=[True, False])
    print(h5_sub.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Generating all figures → figures/")
    print("=" * 60)
    plot_all(results)
    print("Done. Check the figures/ directory.")


if __name__ == '__main__':
    main()
