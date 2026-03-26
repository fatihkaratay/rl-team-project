# Implementation Plan: Adaptive Multi-Site Stimulation Control Using RL

**Project**: EN.705.741.8VL Reinforcement Learning — Semester Project
**Team**: Fatih Karatay & Cody Moxam
**Last Updated**: 2026-03-26

---

## Quick Reference

| Phase | What | Status |
|-------|------|--------|
| 1 | Project scaffold & environment | [x] |
| 2 | MDP environment implementation | [ ] |
| 3 | RL agents implementation | [ ] |
| 4 | Model-based baseline (Value Iteration) | [ ] |
| 5 | Experiment runner & config system | [ ] |
| 6 | Results collection & analysis | [ ] |
| 7 | Visualization & plots | [ ] |
| 8 | Paper writeup support | [ ] |
| 9 | Presentation support | [ ] |

Mark steps `[x]` as you complete them to track progress across sessions.

---

## Project Summary (for context)

Single-agent finite-horizon MDP simulating adaptive brain stimulation. The agent chooses which of 4 sites (S1–S4) to stimulate at each timestep to maximize cumulative EEG reward. The patient transitions stochastically through states (baseline → receptive → non-receptive) based on stimulation history. The agent must learn to manage patient state while exploiting high-reward sites.

**Core MDP:**
- State: `(current_site, patient_state, t)` — fully observable
- Actions: `{S1, S2, S3, S4}`
- Observations: `{favorable, neutral, unfavorable}` → rewards `{+1, 0, -1}`
- Switching penalty: `c_switch ∈ {0, 0.1, 0.25}`
- Episode horizon: `I ∈ {5, 10}`

**Algorithms to implement:**
1. Monte Carlo Control (on-policy, ε-greedy)
2. Q-Learning (off-policy TD)
3. Expected SARSA (on-policy TD)
4. Double Q-Learning (off-policy, reduced maximization bias)
5. Value Iteration (model-based upper-bound baseline)

---

## Phase 1: Project Scaffold

**Goal**: Set up clean, reproducible Python project structure.

### Steps

- [ ] 1.1 Create directory structure:
  ```
  rl-team-project/
  ├── src/
  │   ├── env/
  │   │   ├── __init__.py
  │   │   └── stimulation_env.py       # MDP environment
  │   ├── agents/
  │   │   ├── __init__.py
  │   │   ├── base_agent.py            # Abstract base class
  │   │   ├── monte_carlo.py
  │   │   ├── q_learning.py
  │   │   ├── expected_sarsa.py
  │   │   ├── double_q_learning.py
  │   │   └── value_iteration.py       # Model-based baseline
  │   ├── experiments/
  │   │   ├── __init__.py
  │   │   ├── runner.py                # Run one config
  │   │   └── configs.py               # All experiment configurations
  │   ├── analysis/
  │   │   ├── __init__.py
  │   │   └── metrics.py               # Compute diagnostics
  │   └── visualization/
  │       ├── __init__.py
  │       └── plots.py                 # All plotting functions
  ├── notebooks/
  │   ├── 01_env_sanity_check.ipynb
  │   ├── 02_agent_debugging.ipynb
  │   └── 03_results_analysis.ipynb
  ├── results/                         # Saved experiment outputs (gitignored large files)
  ├── figures/                         # Exported plots for paper/presentation
  ├── docs/
  │   ├── PROPOSAL.md
  │   ├── proposal.pdf
  │   ├── PAPER.md
  │   └── Project_Presentation.pptx
  ├── PLAN.md                          # This file
  ├── README.md
  ├── requirements.txt
  └── .gitignore
  ```

- [ ] 1.2 Create `requirements.txt`:
  ```
  numpy
  matplotlib
  seaborn
  pandas
  jupyter
  tqdm
  scipy
  ```

- [ ] 1.3 Update `.gitignore` to exclude `results/` large pickle files but keep `figures/`.

---

## Phase 2: MDP Environment

**File**: `src/env/stimulation_env.py`
**Goal**: A self-contained, deterministic-seed environment that fully implements the MDP.

### Steps

- [ ] 2.1 Define constants and transition/observation tables as module-level dicts:
  - `SITES = ['S1', 'S2', 'S3', 'S4']`
  - `PATIENT_STATES = ['baseline', 'receptive', 'non_receptive']`
  - `OBSERVATIONS = ['favorable', 'neutral', 'unfavorable']`
  - `REWARD_MAP = {'favorable': +1, 'neutral': 0, 'unfavorable': -1}`
  - Patient transition matrix `T[patient_state][same_site | diff_site]` (probabilities from proposal)
  - Observation model `P_obs[setting][site][patient_state]` for all 3 settings (high/moderate/low separation)

- [ ] 2.2 Implement `StimulationEnv` class:
  ```python
  class StimulationEnv:
      def __init__(self, setting='high', horizon=10, c_switch=0.0, seed=None)
      def reset() -> state        # s0 = ('start', 'baseline', 0)
      def step(action) -> (next_state, reward, done, info)
      def state_space() -> list   # all valid (site, patient_state, t) tuples
      def action_space() -> list  # ['S1', 'S2', 'S3', 'S4']
      def get_transition_probs(state, action) -> dict  # for value iteration
  ```

- [ ] 2.3 Encode state as a tuple `(current_site_idx, patient_state_idx, t)` — use integers internally for Q-table indexing, expose human-readable form via property.

- [ ] 2.4 Implement `get_transition_probs(state, action)` that returns the full distribution over `(next_state, reward, prob)` tuples — needed by Value Iteration.

- [ ] 2.5 **Sanity checks** (in notebook `01_env_sanity_check.ipynb`):
  - Verify transition probabilities sum to 1.0 for all state-action pairs
  - Verify observation probabilities sum to 1.0
  - Run random policy for 1000 episodes; check state distribution is reasonable
  - Verify episode terminates exactly at horizon `I`

---

## Phase 3: RL Agents

**Goal**: All agents share a common interface and work plug-and-play with the environment.

### 3.0 Base Agent (`src/agents/base_agent.py`)

- [ ] 3.0.1 Abstract base class:
  ```python
  class BaseAgent:
      def __init__(self, env, alpha, epsilon, epsilon_decay, gamma=1.0, seed=None)
      def select_action(state) -> action           # ε-greedy
      def update(state, action, reward, next_state, done)
      def train(n_episodes) -> list[float]         # returns episode returns
      def get_Q() -> np.ndarray
      def reset()                                  # reset for new seed run
  ```
- [ ] 3.0.2 Q-table: `np.zeros((n_sites+1, n_patient_states, horizon+1, n_actions))` — indexed by state tuple directly.
- [ ] 3.0.3 ε-greedy with exponential decay: `ε = max(ε_min, ε_0 * decay^episode)`.

### 3.1 Monte Carlo Control (`src/agents/monte_carlo.py`)

- [ ] 3.1.1 On-policy first-visit MC with ε-greedy behavior policy.
- [ ] 3.1.2 Collect full episode trajectory `[(s, a, r), ...]`, then compute `G_t = Σ γ^k r_{t+k}` backwards.
- [ ] 3.1.3 Update `Q(s,a)` using incremental mean (constant step-size `alpha`) for first visits only.
- [ ] 3.1.4 After each episode update, decay ε.

### 3.2 Q-Learning (`src/agents/q_learning.py`)

- [ ] 3.2.1 Off-policy TD(0): `Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]`
- [ ] 3.2.2 Update online after each step.
- [ ] 3.2.3 Behavior policy is ε-greedy; target policy is greedy.

### 3.3 Expected SARSA (`src/agents/expected_sarsa.py`)

- [ ] 3.3.1 `Q(s,a) ← Q(s,a) + α[r + γ·Σ_a' π(a'|s')·Q(s',a') - Q(s,a)]`
- [ ] 3.3.2 Expected value under ε-greedy policy: `(ε/|A|) * sum(Q(s',:)) + (1-ε) * max(Q(s',:])`
- [ ] 3.3.3 Update online after each step.

### 3.4 Double Q-Learning (`src/agents/double_q_learning.py`)

- [ ] 3.4.1 Maintain two Q-tables: `Q1`, `Q2`.
- [ ] 3.4.2 At each step, with 50% probability update `Q1` using `Q2` for bootstrap target, else update `Q2` using `Q1`:
  - Update `Q1`: `Q1(s,a) ← Q1(s,a) + α[r + γ·Q2(s', argmax_a' Q1(s',a')) - Q1(s,a)]`
  - Update `Q2`: symmetric
- [ ] 3.4.3 Action selection: ε-greedy on `(Q1 + Q2) / 2`.

### 3.5 Value Iteration (`src/agents/value_iteration.py`)

- [ ] 3.5.1 Requires full model access — use `env.get_transition_probs()`.
- [ ] 3.5.2 Backward induction for finite horizon: start from `t = I`, work backwards to `t = 0`.
  - `V*(s, I) = 0` for all terminal states
  - `V*(s, t) = max_a Σ P(s',r|s,a) [r + γ V*(s', t+1)]`
- [ ] 3.5.3 Store optimal `Q*(s,a,t)` values for comparison.
- [ ] 3.5.4 This gives the **true optimal policy** — the upper bound for all model-free agents.

---

## Phase 4: Experiment Runner

**File**: `src/experiments/runner.py`, `src/experiments/configs.py`
**Goal**: Systematically run all algorithm × setting × parameter combinations; save structured results.

### Steps

- [ ] 4.1 Define experiment config as a dataclass or dict:
  ```python
  @dataclass
  class ExperimentConfig:
      setting: str          # 'high', 'moderate', 'low'
      horizon: int          # 5 or 10
      c_switch: float       # 0, 0.1, 0.25
      algorithm: str        # 'mc', 'qlearning', 'expected_sarsa', 'double_q', 'value_iter'
      alpha: float          # e.g. 0.1
      epsilon: float        # e.g. 1.0
      epsilon_decay: float  # e.g. 0.995
      n_episodes: int       # e.g. 5000
      n_seeds: int          # e.g. 10
  ```

- [ ] 4.2 Define the **full factorial experiment matrix** in `configs.py`:
  - Settings: `['high', 'moderate', 'low']`
  - Horizons: `[5, 10]`
  - Switch costs: `[0, 0.1, 0.25]`
  - Algorithms: all 5
  - → 3 × 2 × 3 × 5 = **90 experiment configs** (before hyperparameter sensitivity)

- [ ] 4.3 Implement `run_experiment(config) -> ExperimentResult`:
  - For each seed: create env + agent, train for `n_episodes`, collect episode returns + diagnostics
  - Return: mean/std of returns, Q-table, per-episode metrics

- [ ] 4.4 Diagnostics to collect per episode:
  - `episode_return` — total reward
  - `site_visit_counts` — fraction of steps at each site
  - `patient_state_fractions` — fraction of time in each patient state
  - `switch_count` — number of site switches
  - `avg_consecutive_per_site` — mean consecutive stimulations before switching

- [ ] 4.5 Save results as pickle or JSON in `results/` directory. Use config hash as filename.

- [ ] 4.6 Implement `run_all()` with `tqdm` progress bar; supports resume (skip configs already computed).

---

## Phase 5: Analysis & Metrics

**File**: `src/analysis/metrics.py`
**Goal**: Load saved results and compute summary statistics for all hypotheses.

### Steps

- [ ] 5.1 Learning curve smoother: rolling mean over K episodes.
- [ ] 5.2 Convergence detector: episode where return stabilizes within X% of final value.
- [ ] 5.3 Per-hypothesis analysis functions:
  - `H1_convergence_by_setting()`: compare convergence speed across high/moderate/low settings
  - `H2_switching_by_cost()`: switching frequency and non-receptive fraction vs `c_switch`
  - `H3_double_q_stability()`: variance of returns, Q-value magnitude (Q-Learning vs Double Q)
  - `H4_model_free_vs_optimal()`: gap between each agent and Value Iteration upper bound
  - `H5_state_management()`: patient state distribution over time; site alternation patterns
- [ ] 5.4 Summary table generator: algorithm × setting → mean final return ± std.

---

## Phase 6: Visualization

**File**: `src/visualization/plots.py`
**Goal**: Publication-quality figures for paper and slides.

### Planned Figures

- [ ] 6.1 **Learning curves**: Episode return vs episodes for all 5 algorithms (1 plot per setting × horizon; smoothed + confidence bands across seeds)
- [ ] 6.2 **Final performance heatmap**: Algorithm × Setting → mean return (3 × 5 grid)
- [ ] 6.3 **Patient state distribution**: Stacked bar or area chart — fraction of time in each patient state per algorithm
- [ ] 6.4 **Site visit frequency**: Bar chart per algorithm showing which sites are most visited
- [ ] 6.5 **Switching frequency vs switch cost**: Line plot per algorithm
- [ ] 6.6 **Q-value heatmap**: `Q*(s, a)` from Value Iteration — shows optimal policy structure
- [ ] 6.7 **Algorithm gap to optimal**: Bar chart of `V* - V_agent` for each algorithm/setting
- [ ] 6.8 **H3 variance plot**: Box plots of episode returns: Q-Learning vs Double Q-Learning

All figures: save as `figures/<name>.pdf` and `.png` at 300 DPI.

---

## Phase 7: Hyperparameter Sensitivity (Optional / Time Permitting)

- [ ] 7.1 Grid search on `alpha ∈ {0.01, 0.05, 0.1, 0.3}` for Q-Learning in high-separation setting.
- [ ] 7.2 Grid search on `epsilon_decay ∈ {0.99, 0.995, 0.999}`.
- [ ] 7.3 Sensitivity plot: final return vs alpha; identify robust range.
- [ ] 7.4 Use best hyperparams for all main experiments.

---

## Phase 8: Paper Writeup Support

**File**: `docs/PAPER.md`

### Sections to fill in (in order of completion)

- [ ] 8.1 **Methods**: Environment description, MDP formulation, algorithm descriptions
- [ ] 8.2 **Experimental Setup**: Config table, number of seeds, training episodes, hardware
- [ ] 8.3 **Results**: All figures + tables from Phase 6; hypothesis-by-hypothesis findings
- [ ] 8.4 **Discussion**: What did agents learn? Which hypotheses were confirmed? Surprises?
- [ ] 8.5 **Conclusion**: Summary, limitations, future work

---

## Phase 9: Presentation

**File**: `docs/Project_Presentation.pptx`

### Suggested Slide Structure

- [ ] Slide 1: Title + Team
- [ ] Slide 2: Problem Motivation (why brain stimulation as RL?)
- [ ] Slide 3: MDP formulation (state, action, reward — keep concise)
- [ ] Slide 4: Patient State Dynamics (transition table diagram)
- [ ] Slide 5: Algorithms overview (1-liner each + key difference)
- [ ] Slide 6: Experimental design (settings × costs × horizons)
- [ ] Slide 7-10: Results — one slide per hypothesis (figure + finding)
- [ ] Slide 11: Summary table (algorithm performance across settings)
- [ ] Slide 12: Conclusions + Future Work

---

## Implementation Order (Recommended Sequence)

```
Phase 1 (scaffold) → Phase 2 (env) → [sanity check env] →
Phase 3.0 (base agent) → Phase 3.5 (value iteration) → [verify optimal policy] →
Phase 3.2 (Q-Learning) → [compare to VI] →
Phase 3.1 (MC) → Phase 3.3 (Expected SARSA) → Phase 3.4 (Double Q) →
Phase 4 (experiment runner) → Phase 5 (analysis) → Phase 6 (visualization) →
Phase 8 (paper) → Phase 9 (presentation)
```

Value Iteration first (before model-free agents) gives you ground truth to validate that model-free agents are heading in the right direction before running all 90 configs.

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| State encoding | Integer tuple `(site_idx, ps_idx, t)` | Direct Q-table indexing |
| Discount factor | γ = 1.0 | Finite horizon; care about total return |
| Episode return metric | Undiscounted sum | Aligns with win/loss definition in proposal |
| Seeds | 10 seeds per config | Balance statistical reliability vs compute |
| n_episodes | 5000 (TD), 2000 (MC) | MC episodes are longer to converge |
| ε schedule | Exponential decay to ε_min = 0.05 | Ensures continued exploration in early episodes |

---

## Notes / Decisions Log

*(Add dated notes here as the project evolves)*

- **2026-03-26**: Plan created. Starting from Phase 1.
