# Adaptive Multi-Site Stimulation Control Using RL

**EN.705.741.8VL Reinforcement Learning — Semester Project**
**Team**: Fatih Karatay & Cody Moxam

## Overview

A finite-horizon MDP simulating adaptive brain stimulation. An RL agent selects which of 4 stimulation sites (S1–S4) to target at each timestep to maximize cumulative EEG reward, while managing stochastic patient state transitions (baseline → receptive → non-receptive).

See [`docs/PROPOSAL.md`](docs/PROPOSAL.md) for the full MDP specification.
See [`PLAN.md`](PLAN.md) for the implementation plan and progress tracking.

## Project Structure

```
src/
├── env/            # MDP environment (StimulationEnv)
├── agents/         # RL agents (MC, Q-Learning, Expected SARSA, Double Q, Value Iteration)
├── experiments/    # Experiment configs and runner
├── analysis/       # Metrics and hypothesis analysis
└── visualization/  # Plotting functions
notebooks/          # Jupyter notebooks for exploration and analysis
results/            # Saved experiment outputs (gitignored)
figures/            # Exported plots for paper/presentation
docs/               # Proposal, paper draft, presentation
```

## Setup

```bash
pip install -r requirements.txt
```

## Algorithms

- Monte Carlo Control (on-policy ε-greedy)
- Q-Learning (off-policy TD)
- Expected SARSA (on-policy TD)
- Double Q-Learning (reduced maximization bias)
- Value Iteration (model-based upper-bound baseline)
