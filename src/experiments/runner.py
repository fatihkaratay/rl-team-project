"""
Phase 4: Experiment runner

run_experiment(config)  — runs one config over n_seeds, returns ExperimentResult
run_all(configs)        — batch runner with tqdm + resume support
save_result / load_result / result_exists — persistence helpers
"""
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from src.env.stimulation_env import StimulationEnv
from src.agents.monte_carlo import MonteCarloAgent
from src.agents.q_learning import QLearningAgent
from src.agents.expected_sarsa import ExpectedSARSAAgent
from src.agents.double_q_learning import DoubleQLearningAgent
from src.agents.value_iteration import ValueIterationAgent
from .configs import ExperimentConfig

# Results directory relative to project root
_HERE       = os.path.dirname(__file__)                          # src/experiments/
_PROJ_ROOT  = os.path.dirname(os.path.dirname(_HERE))           # project root
RESULTS_DIR = os.path.join(_PROJ_ROOT, 'results')


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """
    All per-seed, per-episode data for one ExperimentConfig.

    Array shapes
    ------------
    returns          : (n_seeds, n_episodes)
    site_visit_fracs : (n_seeds, n_episodes, 4)   — fraction at S1-S4 per episode
    ps_fracs         : (n_seeds, n_episodes, 3)   — fraction in each patient state
    switch_counts    : (n_seeds, n_episodes)       — #site switches per episode
    avg_consecutive  : (n_seeds, n_episodes)       — mean consecutive-site run length
    q_tables         : list of n_seeds arrays      — final Q-tables
    """
    config           : ExperimentConfig
    returns          : np.ndarray
    site_visit_fracs : np.ndarray
    ps_fracs         : np.ndarray
    switch_counts    : np.ndarray
    avg_consecutive  : np.ndarray
    q_tables         : List[np.ndarray]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_diagnostics(steps: list) -> dict:
    """
    Compute per-episode diagnostics from a list of next_state tuples.

    steps : [(site_idx, ps_idx, t), ...]  — one entry per timestep

    Returns
    -------
    dict with keys: site_fracs, ps_fracs, switch_count, avg_consecutive
    """
    n             = len(steps)
    site_counts   = np.zeros(4)
    ps_counts     = np.zeros(3)
    switch_count  = 0
    runs: List[int] = []
    current_run   = 0
    prev_site     = None

    for site, ps, _ in steps:
        # site is always 1-4 after the first action (Start=0 only at t=0)
        if site > 0:
            site_counts[site - 1] += 1
        ps_counts[ps] += 1

        if prev_site is None:
            current_run = 1
        elif site == prev_site:
            current_run += 1
        else:
            switch_count += 1
            runs.append(current_run)
            current_run = 1
        prev_site = site

    if current_run > 0:
        runs.append(current_run)

    denom = n if n > 0 else 1
    return {
        'site_fracs'     : site_counts / denom,
        'ps_fracs'       : ps_counts   / denom,
        'switch_count'   : switch_count,
        'avg_consecutive': float(np.mean(runs)) if runs else 0.0,
    }


def _run_model_free_episode(agent, env) -> tuple:
    """
    Run one training episode for a model-free agent.

    For TD agents (Q-Learning, Expected SARSA, Double Q): update at every step.
    For Monte Carlo: collect full trajectory, then apply first-visit update.

    Returns (episode_return: float, diagnostics: dict).
    """
    is_mc  = isinstance(agent, MonteCarloAgent)
    state  = env.reset()
    done   = False
    ep_ret = 0.0
    steps  : List[tuple] = []   # next_states for diagnostics
    traj   : List[tuple] = []   # (s, a, r) for MC update

    while not done:
        action               = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        ep_ret              += reward
        steps.append(next_state)

        if is_mc:
            traj.append((state, action, reward))
        else:
            agent.update(state, action, reward, next_state, done)

        state = next_state

    # MC end-of-episode first-visit update
    if is_mc:
        G        = 0.0
        visited: set = set()
        for s, a, r in reversed(traj):
            G = r + agent.gamma * G
            if (s, a) not in visited:
                visited.add((s, a))
                si, pi, ti = s
                agent.Q[si, pi, ti, a] += agent.alpha * (G - agent.Q[si, pi, ti, a])

    agent._decay_epsilon()
    return ep_ret, _compute_diagnostics(steps)


def _run_vi_episode(vi: ValueIterationAgent, env) -> tuple:
    """
    Run one evaluation episode with the optimal VI policy.
    Returns (episode_return: float, diagnostics: dict).
    """
    state  = env.reset()
    done   = False
    ep_ret = 0.0
    steps  : List[tuple] = []

    while not done:
        action               = vi.get_action(state)
        next_state, reward, done, _ = env.step(action)
        ep_ret              += reward
        steps.append(next_state)
        state = next_state

    return ep_ret, _compute_diagnostics(steps)


def _make_agent(config: ExperimentConfig, env, seed: int):
    """Instantiate the model-free agent specified by config.algorithm."""
    kwargs = dict(
        alpha         = config.alpha,
        epsilon       = config.epsilon,
        epsilon_decay = config.epsilon_decay,
        epsilon_min   = config.epsilon_min,
        gamma         = 1.0,
        seed          = seed,
    )
    _map = {
        'mc'            : MonteCarloAgent,
        'qlearning'     : QLearningAgent,
        'expected_sarsa': ExpectedSARSAAgent,
        'double_q'      : DoubleQLearningAgent,
    }
    return _map[config.algorithm](env, **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Run *config* across all seeds and return an ExperimentResult.

    For model-free algorithms: trains for config.n_episodes per seed.
    For Value Iteration: solves once then evaluates for config.n_episodes episodes.
    """
    n_ep    = config.n_episodes
    n_seeds = config.n_seeds

    returns          = np.zeros((n_seeds, n_ep))
    site_visit_fracs = np.zeros((n_seeds, n_ep, 4))
    ps_fracs_arr     = np.zeros((n_seeds, n_ep, 3))
    switch_counts    = np.zeros((n_seeds, n_ep))
    avg_consecutive  = np.zeros((n_seeds, n_ep))
    q_tables         = []

    for s in range(n_seeds):
        env = StimulationEnv(
            setting  = config.setting,
            horizon  = config.horizon,
            c_switch = config.c_switch,
            seed     = s,
        )

        if config.algorithm == 'value_iter':
            vi = ValueIterationAgent(env, gamma=1.0)
            vi.solve()
            for ep in range(n_ep):
                r, d                        = _run_vi_episode(vi, env)
                returns[s, ep]              = r
                site_visit_fracs[s, ep]     = d['site_fracs']
                ps_fracs_arr[s, ep]         = d['ps_fracs']
                switch_counts[s, ep]        = d['switch_count']
                avg_consecutive[s, ep]      = d['avg_consecutive']
            q_tables.append(vi.get_Q())
        else:
            agent = _make_agent(config, env, seed=s * 1000)
            for ep in range(n_ep):
                r, d                        = _run_model_free_episode(agent, env)
                returns[s, ep]              = r
                site_visit_fracs[s, ep]     = d['site_fracs']
                ps_fracs_arr[s, ep]         = d['ps_fracs']
                switch_counts[s, ep]        = d['switch_count']
                avg_consecutive[s, ep]      = d['avg_consecutive']
            q_tables.append(agent.get_Q())

    return ExperimentResult(
        config           = config,
        returns          = returns,
        site_visit_fracs = site_visit_fracs,
        ps_fracs         = ps_fracs_arr,
        switch_counts    = switch_counts,
        avg_consecutive  = avg_consecutive,
        q_tables         = q_tables,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _result_path(config: ExperimentConfig) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"{config.config_hash()}.pkl")


def save_result(result: ExperimentResult) -> str:
    """Pickle *result* to results/<hash>.pkl.  Returns the file path."""
    path = _result_path(result.config)
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    return path


def load_result(config: ExperimentConfig) -> ExperimentResult:
    """Load the saved result for *config*."""
    with open(_result_path(config), 'rb') as f:
        return pickle.load(f)


def result_exists(config: ExperimentConfig) -> bool:
    """Return True if a saved result file exists for *config*."""
    return os.path.isfile(_result_path(config))


def run_all(configs=None) -> list:
    """
    Run (or resume) all configs.

    Already-computed configs are loaded from disk; new ones are run and saved.
    Returns a list of ExperimentResult in the same order as *configs*.
    """
    from .configs import get_all_configs
    if configs is None:
        configs = get_all_configs()

    results = []
    for config in tqdm(configs, desc='Experiments'):
        if result_exists(config):
            results.append(load_result(config))
        else:
            result = run_experiment(config)
            save_result(result)
            results.append(result)
    return results
