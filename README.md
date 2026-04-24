# Adaptive Multi-Site Stimulation Control Using Reinforcement Learning

**EN.705.741.8VL Reinforcement Learning — Final Team Project**
**Authors:** Fatih Karatay & Cody Moxam
**Term:** Spring 2026

## Overview

We model adaptive brain stimulation as a finite-horizon Markov decision process. At each timestep, an agent selects one of four stimulation sites (S1–S4). The observed EEG response and reward depend on both the chosen site and a stochastic patient-response state (`baseline`, `receptive`, `non-receptive`). Repeated stimulation of the same site biases the patient toward `non-receptive`; switching aids recovery but may incur a switching cost. The framing tests whether site selection is better posed as sequential control rather than static arm selection.

Five algorithms are evaluated across a 90-configuration sweep (3 response-separation levels × 2 horizons × 3 switching costs × 5 algorithms, 10 seeds each):

- Monte Carlo Control (on-policy, ε-greedy)
- Q-Learning (off-policy TD)
- Expected SARSA (on-policy TD)
- Double Q-Learning (reduced maximization bias)
- Value Iteration (model-based upper-bound benchmark)

## Deliverables

| Artifact | Location |
|---|---|
| MDP specification | [`docs/PROPOSAL.md`](docs/PROPOSAL.md) |
| Final paper (PDF) | [`docs/PAPER_V2.pdf`](docs/PAPER_V2.pdf) |
| Final paper (markdown source) | [`docs/PAPER_v2.md`](docs/PAPER_v2.md) |
| Presentation deck | [`docs/Adaptive_Multi_Site_RL_Locked_Results_Deck.pptx`](docs/Adaptive_Multi_Site_RL_Locked_Results_Deck.pptx) |
| Presentation script | [`docs/PRESENTATION_SCRIPT.md`](docs/PRESENTATION_SCRIPT.md) |
| Generated figures (PDF + PNG, 300 DPI) | [`figures/`](figures/) |
| Implementation plan | [`PLAN.md`](PLAN.md) |

The figures in `figures/` are committed, so the paper and slides can be reviewed without re-running any experiments.

## Project Structure

```
src/
├── env/            # MDP environment (StimulationEnv)
├── agents/         # 5 agents: MC, Q-Learning, Expected SARSA, Double Q, Value Iteration
├── experiments/    # Configs + experiment runner (resume-safe)
├── analysis/       # H1–H5 hypothesis tests and summary metrics
└── visualization/  # Figure generation
tests/              # pytest suite (Phases 1–4)
notebooks/          # Environment sanity-check notebook
results/            # Experiment pickles (gitignored — regenerate via run_experiments.py)
figures/            # Committed figures used in paper and slides
docs/               # Proposal, paper, presentation, feedback
```

## Setup

Requires **Python 3.10 or newer** (developed against 3.13).

```bash
# 1. Clone or unzip the repo, then cd into it
cd rl-team-project

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt
```

## Reproducing the Results

All commands assume the virtualenv is active and you are at the repo root.

### 1. Smoke test (≈10 seconds)

Verifies the environment, agents, and pipeline end-to-end on a tiny config.

```bash
python run_all.py --smoke
```

### 2. Full pipeline (≈10–20 minutes on a laptop)

Runs all 90 experiment configurations, prints H1–H5 summary tables, and regenerates every figure in `figures/`. The runner is **resume-safe**: any results already present in `results/` are skipped, so re-running picks up where it left off.

```bash
python run_all.py
```

### 3. Run experiments only

```bash
python run_experiments.py
```

### 4. Re-generate analysis tables and figures from existing pickles

If `results/` already contains the experiment pickles, this skips simulation and just reproduces the tables and figures.

```bash
python run_analysis.py
```

### 5. Test suite

```bash
pytest
```

### 6. Environment sanity-check notebook (optional)

```bash
jupyter notebook notebooks/01_env_sanity_check.ipynb
```

## Key Findings

- **H1 (response separation → performance):** Higher separation yields higher mean final return across all algorithms.
- **H2 (switching cost tradeoff):** Increasing `c_switch` reduces switching frequency (Spearman ρ ≈ −0.83 to −0.94) but raises non-receptive state occupancy (ρ ≈ 0.76 to 0.85).
- **H3 (Double Q stability — null finding):** Double Q-Learning did **not** show significantly lower variance than Q-Learning (Levene's test, p > 0.65 in all settings).
- **H4 (VI upper bound):** Value Iteration outperforms model-free methods in the high- and moderate-separation settings (p < 0.05).
- **H5 (structured policies):** Converged policies show site preferences aligned with the reward model rather than uniform selection.
- Expected SARSA is the strongest model-free baseline; Monte Carlo Control consistently underperforms TD methods (p < 0.05).

See [`docs/PAPER_V2.pdf`](docs/PAPER_V2.pdf) for the full write-up.
