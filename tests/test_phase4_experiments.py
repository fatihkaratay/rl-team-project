"""
Phase 4 tests: experiment configuration, runner, diagnostics, and persistence.

Sections
--------
1.  ExperimentConfig — creation, validation, hashing
2.  get_all_configs() — factorial completeness
3.  _compute_diagnostics — correctness of per-episode metrics
4.  run_experiment() — result shapes and value ranges for all 5 algorithms
5.  Diagnostics: site_fracs and ps_fracs sum to 1
6.  Diagnostics: switch_count and avg_consecutive edge cases
7.  Persistence: save / load / result_exists
8.  run_all() — subset execution and resume support
"""
import os
import numpy as np
import pytest

from src.experiments.configs import (
    ExperimentConfig, get_all_configs,
    VALID_ALGORITHMS, VALID_SETTINGS,
)
from src.experiments.runner import (
    ExperimentResult, run_experiment,
    save_result, load_result, result_exists,
    run_all, _compute_diagnostics, RESULTS_DIR,
)
from src.env.stimulation_env import N_ACTIONS, N_PATIENT_STATES, N_SITES_STATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fast_config(algorithm='qlearning', n_episodes=5, n_seeds=2, **kwargs) -> ExperimentConfig:
    """Minimal config for fast tests."""
    defaults = dict(
        setting='high', horizon=5, c_switch=0.0,
        algorithm=algorithm, n_episodes=n_episodes, n_seeds=n_seeds,
    )
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)


# ===========================================================================
# 1. ExperimentConfig
# ===========================================================================

def test_config_creation():
    cfg = ExperimentConfig(setting='high', horizon=5, c_switch=0.0, algorithm='qlearning')
    assert cfg.setting == 'high'
    assert cfg.horizon == 5


@pytest.mark.parametrize("setting", VALID_SETTINGS)
def test_config_valid_settings(setting):
    cfg = ExperimentConfig(setting=setting, horizon=5, c_switch=0.0, algorithm='qlearning')
    assert cfg.setting == setting


@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_config_valid_algorithms(algorithm):
    cfg = ExperimentConfig(setting='high', horizon=5, c_switch=0.0, algorithm=algorithm)
    assert cfg.algorithm == algorithm


def test_config_invalid_setting_raises():
    with pytest.raises(ValueError, match="setting"):
        ExperimentConfig(setting='extreme', horizon=5, c_switch=0.0, algorithm='qlearning')


def test_config_invalid_algorithm_raises():
    with pytest.raises(ValueError, match="algorithm"):
        ExperimentConfig(setting='high', horizon=5, c_switch=0.0, algorithm='ppo')


def test_config_invalid_horizon_raises():
    with pytest.raises(ValueError, match="horizon"):
        ExperimentConfig(setting='high', horizon=0, c_switch=0.0, algorithm='qlearning')


def test_config_invalid_c_switch_raises():
    with pytest.raises(ValueError, match="c_switch"):
        ExperimentConfig(setting='high', horizon=5, c_switch=-0.1, algorithm='qlearning')


def test_config_invalid_n_episodes_raises():
    with pytest.raises(ValueError, match="n_episodes"):
        ExperimentConfig(setting='high', horizon=5, c_switch=0.0, algorithm='qlearning', n_episodes=0)


def test_config_hash_is_string():
    cfg = fast_config()
    h = cfg.config_hash()
    assert isinstance(h, str) and len(h) == 12


def test_config_hash_deterministic():
    cfg1 = fast_config()
    cfg2 = fast_config()
    assert cfg1.config_hash() == cfg2.config_hash()


def test_config_hash_differs_for_different_configs():
    cfg1 = fast_config(setting='high')
    cfg2 = fast_config(setting='low')
    assert cfg1.config_hash() != cfg2.config_hash()


def test_config_hash_sensitive_to_all_fields():
    base = fast_config()
    variants = [
        fast_config(horizon=10),
        fast_config(c_switch=0.1),
        fast_config(algorithm='mc'),
        fast_config(n_seeds=5),
    ]
    hashes = {v.config_hash() for v in variants}
    assert len(hashes) == len(variants), "Each variant should have a unique hash"


# ===========================================================================
# 2. get_all_configs()
# ===========================================================================

def test_get_all_configs_count():
    configs = get_all_configs()
    assert len(configs) == 90, f"Expected 90 configs, got {len(configs)}"


def test_get_all_configs_covers_all_settings():
    configs = get_all_configs()
    settings = {c.setting for c in configs}
    assert settings == set(VALID_SETTINGS)


def test_get_all_configs_covers_all_algorithms():
    configs = get_all_configs()
    algorithms = {c.algorithm for c in configs}
    assert algorithms == set(VALID_ALGORITHMS)


def test_get_all_configs_covers_all_horizons():
    configs = get_all_configs()
    horizons = {c.horizon for c in configs}
    assert horizons == {5, 10}


def test_get_all_configs_covers_all_c_switch():
    configs = get_all_configs()
    costs = {c.c_switch for c in configs}
    assert costs == {0.0, 0.1, 0.25}


def test_get_all_configs_no_duplicates():
    configs = get_all_configs()
    hashes = [c.config_hash() for c in configs]
    assert len(hashes) == len(set(hashes)), "Duplicate configs detected"


def test_get_all_configs_mc_fewer_episodes():
    configs = get_all_configs()
    mc_eps = {c.n_episodes for c in configs if c.algorithm == 'mc'}
    td_eps = {c.n_episodes for c in configs if c.algorithm == 'qlearning'}
    assert max(mc_eps) < max(td_eps), "MC should have fewer episodes than TD"


def test_get_all_configs_returns_experiment_config_instances():
    configs = get_all_configs()
    assert all(isinstance(c, ExperimentConfig) for c in configs)


# ===========================================================================
# 3. _compute_diagnostics
# ===========================================================================

def test_diagnostics_single_site_no_switch():
    """Always at S1 (site_idx=1) — no switches, avg_consecutive = horizon."""
    steps = [(1, 0, t) for t in range(1, 6)]  # 5 steps at S1
    d = _compute_diagnostics(steps)
    assert d['switch_count'] == 0
    assert abs(d['avg_consecutive'] - 5.0) < 1e-9
    assert abs(d['site_fracs'][0] - 1.0) < 1e-9      # all at S1
    assert abs(d['site_fracs'][1:].sum()) < 1e-9
    assert abs(d['ps_fracs'].sum() - 1.0) < 1e-9


def test_diagnostics_alternating_sites():
    """Alternating S1/S2 every step — switches = horizon-1, avg_consecutive = 1."""
    steps = [(1 + i % 2, 0, i) for i in range(1, 6)]   # S1,S2,S1,S2,S1
    d = _compute_diagnostics(steps)
    assert d['switch_count'] == 4
    assert abs(d['avg_consecutive'] - 1.0) < 1e-9


def test_diagnostics_site_fracs_sum_to_one():
    steps = [(1, 0, 1), (2, 1, 2), (3, 2, 3), (4, 0, 4), (1, 0, 5)]
    d = _compute_diagnostics(steps)
    assert abs(d['site_fracs'].sum() - 1.0) < 1e-9


def test_diagnostics_ps_fracs_sum_to_one():
    steps = [(1, 0, 1), (1, 1, 2), (1, 2, 3)]
    d = _compute_diagnostics(steps)
    assert abs(d['ps_fracs'].sum() - 1.0) < 1e-9


def test_diagnostics_empty_steps():
    """Empty episode should not crash."""
    d = _compute_diagnostics([])
    assert d['switch_count'] == 0
    assert d['avg_consecutive'] == 0.0


def test_diagnostics_all_patient_states():
    steps = [(1, 0, 1), (1, 1, 2), (1, 2, 3)]
    d = _compute_diagnostics(steps)
    assert abs(d['ps_fracs'][0] - 1/3) < 1e-9
    assert abs(d['ps_fracs'][1] - 1/3) < 1e-9
    assert abs(d['ps_fracs'][2] - 1/3) < 1e-9


# ===========================================================================
# 4. run_experiment() — shapes and value ranges
# ===========================================================================

@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_run_experiment_result_type(algorithm):
    cfg = fast_config(algorithm=algorithm)
    result = run_experiment(cfg)
    assert isinstance(result, ExperimentResult)


@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_run_experiment_returns_shape(algorithm):
    cfg = fast_config(algorithm=algorithm, n_episodes=4, n_seeds=2)
    result = run_experiment(cfg)
    assert result.returns.shape == (2, 4)


@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_run_experiment_site_fracs_shape(algorithm):
    cfg = fast_config(algorithm=algorithm, n_episodes=4, n_seeds=2)
    result = run_experiment(cfg)
    assert result.site_visit_fracs.shape == (2, 4, 4)


@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_run_experiment_ps_fracs_shape(algorithm):
    cfg = fast_config(algorithm=algorithm, n_episodes=4, n_seeds=2)
    result = run_experiment(cfg)
    assert result.ps_fracs.shape == (2, 4, 3)


@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_run_experiment_switch_counts_shape(algorithm):
    cfg = fast_config(algorithm=algorithm, n_episodes=4, n_seeds=2)
    result = run_experiment(cfg)
    assert result.switch_counts.shape == (2, 4)


@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_run_experiment_avg_consecutive_shape(algorithm):
    cfg = fast_config(algorithm=algorithm, n_episodes=4, n_seeds=2)
    result = run_experiment(cfg)
    assert result.avg_consecutive.shape == (2, 4)


@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_run_experiment_q_tables_count(algorithm):
    cfg = fast_config(algorithm=algorithm, n_episodes=4, n_seeds=3)
    result = run_experiment(cfg)
    assert len(result.q_tables) == 3


@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_run_experiment_q_tables_shape(algorithm):
    cfg = fast_config(algorithm=algorithm, n_episodes=4, n_seeds=2, horizon=5)
    result = run_experiment(cfg)
    expected = (N_SITES_STATE, N_PATIENT_STATES, cfg.horizon + 1, N_ACTIONS)
    for qt in result.q_tables:
        assert qt.shape == expected


def test_run_experiment_returns_finite():
    for algorithm in VALID_ALGORITHMS:
        cfg = fast_config(algorithm=algorithm)
        result = run_experiment(cfg)
        assert np.all(np.isfinite(result.returns))


# ===========================================================================
# 5. Diagnostics value ranges
# ===========================================================================

@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_site_fracs_sum_to_one(algorithm):
    cfg = fast_config(algorithm=algorithm, n_episodes=5)
    result = run_experiment(cfg)
    sums = result.site_visit_fracs.sum(axis=-1)   # (n_seeds, n_episodes)
    assert np.allclose(sums, 1.0, atol=1e-9), f"{algorithm}: site_fracs don't sum to 1"


@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_ps_fracs_sum_to_one(algorithm):
    cfg = fast_config(algorithm=algorithm, n_episodes=5)
    result = run_experiment(cfg)
    sums = result.ps_fracs.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-9), f"{algorithm}: ps_fracs don't sum to 1"


@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_switch_counts_non_negative(algorithm):
    cfg = fast_config(algorithm=algorithm, n_episodes=5)
    result = run_experiment(cfg)
    assert np.all(result.switch_counts >= 0)


@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_switch_counts_bounded_by_horizon(algorithm):
    horizon = 5
    cfg = fast_config(algorithm=algorithm, n_episodes=5, horizon=horizon)
    result = run_experiment(cfg)
    assert np.all(result.switch_counts <= horizon - 1)


@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_avg_consecutive_at_least_one(algorithm):
    cfg = fast_config(algorithm=algorithm, n_episodes=5)
    result = run_experiment(cfg)
    assert np.all(result.avg_consecutive >= 1.0)


@pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
def test_returns_bounded(algorithm):
    horizon, c_switch = 5, 0.25
    cfg = fast_config(algorithm=algorithm, n_episodes=5,
                      horizon=horizon, c_switch=c_switch)
    result = run_experiment(cfg)
    lower = -(horizon * (1 + c_switch))
    upper = horizon
    assert np.all(result.returns >= lower - 1e-9)
    assert np.all(result.returns <= upper + 1e-9)


# ===========================================================================
# 6. switch_count / avg_consecutive edge cases via run_experiment
# ===========================================================================

def test_no_switch_cost_config_runs():
    """c_switch=0 configs should run cleanly."""
    cfg = fast_config(c_switch=0.0)
    result = run_experiment(cfg)
    assert result.returns.shape[0] == cfg.n_seeds


def test_switch_cost_config_runs():
    cfg = fast_config(c_switch=0.25)
    result = run_experiment(cfg)
    assert np.all(np.isfinite(result.returns))


def test_horizon_one_single_step():
    """Horizon=1 means one step per episode — no switches possible."""
    cfg = fast_config(n_episodes=5, horizon=1)
    result = run_experiment(cfg)
    assert np.all(result.switch_counts == 0)
    assert np.allclose(result.avg_consecutive, 1.0)


# ===========================================================================
# 7. Persistence
# ===========================================================================

def test_result_not_exists_before_save(tmp_path, monkeypatch):
    monkeypatch.setattr('src.experiments.runner.RESULTS_DIR', str(tmp_path))
    cfg = fast_config()
    assert not result_exists(cfg)


def test_save_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr('src.experiments.runner.RESULTS_DIR', str(tmp_path))
    cfg = fast_config()
    result = run_experiment(cfg)
    path = save_result(result)
    assert os.path.isfile(path)


def test_result_exists_after_save(tmp_path, monkeypatch):
    monkeypatch.setattr('src.experiments.runner.RESULTS_DIR', str(tmp_path))
    cfg = fast_config()
    result = run_experiment(cfg)
    save_result(result)
    assert result_exists(cfg)


def test_load_result_matches_original(tmp_path, monkeypatch):
    monkeypatch.setattr('src.experiments.runner.RESULTS_DIR', str(tmp_path))
    cfg = fast_config()
    original = run_experiment(cfg)
    save_result(original)
    loaded = load_result(cfg)
    assert np.array_equal(original.returns, loaded.returns)
    assert original.config == loaded.config


def test_save_uses_config_hash_as_filename(tmp_path, monkeypatch):
    monkeypatch.setattr('src.experiments.runner.RESULTS_DIR', str(tmp_path))
    cfg = fast_config()
    result = run_experiment(cfg)
    path = save_result(result)
    assert cfg.config_hash() in os.path.basename(path)


def test_different_configs_save_to_different_files(tmp_path, monkeypatch):
    monkeypatch.setattr('src.experiments.runner.RESULTS_DIR', str(tmp_path))
    cfg1 = fast_config(setting='high')
    cfg2 = fast_config(setting='low')
    save_result(run_experiment(cfg1))
    save_result(run_experiment(cfg2))
    files = os.listdir(tmp_path)
    assert len(files) == 2


# ===========================================================================
# 8. run_all() — subset and resume
# ===========================================================================

def test_run_all_returns_correct_count(tmp_path, monkeypatch):
    monkeypatch.setattr('src.experiments.runner.RESULTS_DIR', str(tmp_path))
    configs = [fast_config(setting=s) for s in ('high', 'moderate', 'low')]
    results = run_all(configs)
    assert len(results) == 3


def test_run_all_returns_experiment_results(tmp_path, monkeypatch):
    monkeypatch.setattr('src.experiments.runner.RESULTS_DIR', str(tmp_path))
    configs = [fast_config()]
    results = run_all(configs)
    assert isinstance(results[0], ExperimentResult)


def test_run_all_resume_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setattr('src.experiments.runner.RESULTS_DIR', str(tmp_path))
    cfg = fast_config()

    # Pre-save a result
    original = run_experiment(cfg)
    save_result(original)

    # Modify the original to detect if run_experiment is called again
    original.returns[:] = 999.0
    save_result(original)   # overwrite with sentinel value

    # run_all should load the saved version (sentinel), not re-run
    results = run_all([cfg])
    assert np.all(results[0].returns == 999.0), "run_all should skip existing results"


def test_run_all_empty_configs(tmp_path, monkeypatch):
    monkeypatch.setattr('src.experiments.runner.RESULTS_DIR', str(tmp_path))
    results = run_all([])
    assert results == []
