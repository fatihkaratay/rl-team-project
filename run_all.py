"""
Master pipeline: runs experiments, computes metrics, and generates all figures.

Usage:
    python run_all.py           # full pipeline (resume-safe)
    python run_all.py --smoke   # quick smoke test only
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from src.experiments.runner import run_all, run_experiment
from src.experiments.configs import get_all_configs, ExperimentConfig
from src.analysis.metrics import (
    compute_summary,
    H1_convergence_by_setting,
    H2_switching_by_cost,
    H3_double_q_stability,
    H4_model_free_vs_optimal,
    H5_state_management,
)
from src.visualization.plots import plot_all

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
pd.set_option('display.float_format', '{:.3f}'.format)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test():
    print("Running smoke test (1 config × 2 seeds × 50 episodes)...")
    config = ExperimentConfig(
        setting='high', horizon=10, c_switch=0.0,
        algorithm='qlearning', n_episodes=50, n_seeds=2,
    )
    result = run_experiment(config)
    print(f"  Q-Learning mean return (last 10 eps): {result.returns[:, -10:].mean():.3f}")
    config_vi = ExperimentConfig(
        setting='high', horizon=10, c_switch=0.0,
        algorithm='value_iter', n_episodes=50, n_seeds=2,
    )
    result_vi = run_experiment(config_vi)
    print(f"  Value Iteration mean return:          {result_vi.returns.mean():.3f}")
    print("Smoke test passed.\n")


# ---------------------------------------------------------------------------
# Step 1: Experiments
# ---------------------------------------------------------------------------

def step_experiments():
    configs = get_all_configs()
    print(f"[1/3] Running experiments ({len(configs)} configs, resume-safe)...")
    results = run_all(configs)
    print(f"      Done. {len(results)} results in results/\n")
    return results


# ---------------------------------------------------------------------------
# Step 2: Metrics
# ---------------------------------------------------------------------------

def step_metrics(results):
    print("[2/3] Computing metrics...\n")

    print("=" * 60)
    print("SUMMARY: Mean Final Return ± 95% bootstrap CI  (horizon=10, c_switch=0)")
    print("=" * 60)
    df = compute_summary(results)
    subset = df[(df.horizon == 10) & (df.c_switch == 0.0)][
        ['algo_label', 'setting', 'mean_return', 'sem', 'ci_low', 'ci_high']
    ].sort_values(['setting', 'mean_return'], ascending=[True, False])
    print(subset.to_string(index=False))
    print()

    print("=" * 60)
    print("H1: Convergence speed by setting (horizon=10, c_switch=0)")
    print("=" * 60)
    h1 = H1_convergence_by_setting(results)
    h1_sub = h1[(h1.horizon == 10) & (h1.c_switch == 0.0)][
        ['algo_label', 'setting', 'convergence_episode', 'final_return']
    ].sort_values(['setting', 'convergence_episode'])
    print(h1_sub.to_string(index=False))
    print()

    print("=" * 60)
    print("H2: Switching behaviour vs switch cost (high setting, horizon=10)")
    print("=" * 60)
    h2 = H2_switching_by_cost(results)
    h2_sub = h2[(h2.setting == 'high') & (h2.horizon == 10)][
        ['algo_label', 'c_switch', 'switch_rate', 'nonreceptive_frac', 'final_return']
    ].sort_values(['algo_label', 'c_switch'])
    print(h2_sub.to_string(index=False))
    print()

    print("=" * 60)
    print("H3: Q-Learning vs Double Q stability (horizon=10, c_switch=0)")
    print("=" * 60)
    h3 = H3_double_q_stability(results)
    h3_sub = h3[(h3.horizon == 10) & (h3.c_switch == 0.0)][
        ['algo_label', 'setting', 'return_variance', 'mean_return']
    ].sort_values(['setting', 'algo_label'])
    print(h3_sub.to_string(index=False))
    print()

    print("=" * 60)
    print("H4: Gap to Value Iteration (horizon=10, c_switch=0)")
    print("=" * 60)
    h4 = H4_model_free_vs_optimal(results)
    h4_sub = h4[(h4.horizon == 10) & (h4.c_switch == 0.0)][
        ['algo_label', 'setting', 'agent_return', 'vi_return', 'gap', 'gap_pct']
    ].sort_values(['setting', 'gap'])
    print(h4_sub.to_string(index=False))
    print()

    print("=" * 60)
    print("H5: Patient state management (horizon=10, c_switch=0)")
    print("=" * 60)
    h5 = H5_state_management(results)
    h5_sub = h5[(h5.horizon == 10) & (h5.c_switch == 0.0)][
        ['algo_label', 'setting', 'early_receptive', 'late_receptive', 'receptive_improvement']
    ].sort_values(['setting', 'receptive_improvement'], ascending=[True, False])
    print(h5_sub.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Step 3: Figures
# ---------------------------------------------------------------------------

def step_figures(results):
    print("[3/3] Generating figures → figures/...")
    plot_all(results)
    print("      Done.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full RL experiment pipeline")
    parser.add_argument('--smoke', action='store_true',
                        help='Quick smoke test only (does not save results)')
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
        return

    results = step_experiments()
    step_metrics(results)
    step_figures(results)
    print("All done.")


if __name__ == '__main__':
    main()
