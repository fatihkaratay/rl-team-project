"""
Entry point for running all 90 experiments.

Usage:
    python run_experiments.py              # run all 90 configs (resume-safe)
    python run_experiments.py --smoke      # quick smoke test: 1 config, 2 seeds, 50 eps
"""
import argparse
import sys
import os

# Make sure src/ is importable from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.experiments.runner import run_all, run_experiment, save_result
from src.experiments.configs import get_all_configs, ExperimentConfig


def smoke_test():
    """Run a single small config to verify everything works end-to-end."""
    print("Running smoke test (1 config × 2 seeds × 50 episodes)...")
    config = ExperimentConfig(
        setting='high', horizon=10, c_switch=0.0,
        algorithm='qlearning', n_episodes=50, n_seeds=2,
    )
    result = run_experiment(config)
    print(f"  Q-Learning mean return (last 10 eps): "
          f"{result.returns[:, -10:].mean():.3f}")

    config_vi = ExperimentConfig(
        setting='high', horizon=10, c_switch=0.0,
        algorithm='value_iter', n_episodes=50, n_seeds=2,
    )
    result_vi = run_experiment(config_vi)
    print(f"  Value Iteration mean return:          "
          f"{result_vi.returns.mean():.3f}")
    print("Smoke test passed.")


def main():
    parser = argparse.ArgumentParser(description="Run RL stimulation experiments")
    parser.add_argument('--smoke', action='store_true',
                        help='Quick smoke test only (does not save results)')
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
        return

    configs = get_all_configs()
    print(f"Running {len(configs)} experiment configs "
          f"(resume-safe: already-computed configs are skipped).")
    results = run_all(configs)
    print(f"\nDone. {len(results)} results saved to results/")


if __name__ == '__main__':
    main()
