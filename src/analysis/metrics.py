"""
Phase 5: Analysis & Metrics

Functions for analyzing ExperimentResult objects and testing project hypotheses H1–H5.
All functions accept the list of ExperimentResult objects returned by run_all().
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALGO_LABELS = {
    'mc':             'Monte Carlo',
    'qlearning':      'Q-Learning',
    'expected_sarsa': 'Expected SARSA',
    'double_q':       'Double Q-Learning',
    'value_iter':     'Value Iteration',
}

MODEL_FREE_ALGOS = ['mc', 'qlearning', 'expected_sarsa', 'double_q']
PATIENT_STATE_LABELS = ['baseline', 'receptive', 'non_receptive']
SITE_LABELS = ['S1', 'S2', 'S3', 'S4']


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def smooth(values, window=50):
    """
    Rolling mean with expanding window at the start (no NaNs).

    Parameters
    ----------
    values : array-like, shape (n,)
    window : int

    Returns
    -------
    np.ndarray, shape (n,)
    """
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=window, min_periods=1)
        .mean()
        .values
    )


def convergence_episode(mean_returns, window=200, k_std=2.0):
    """
    First episode where the heavily-smoothed return enters and stays within
    a noise-band of the final smoothed value.

    The tolerance band is `k_std * tail_std`, where `tail_std` is the std of
    the smoothed curve over the last 25 % of episodes — i.e. the steady-state
    noise level. This makes the metric robust to per-episode variance and
    actually distinguishes algorithms that stabilize at different times.

    Returns len(mean_returns) if convergence is never detected.
    """
    smoothed  = smooth(mean_returns, window=window)
    final_val = smoothed[-1]
    tail_n    = max(1, len(smoothed) // 4)
    tail_std  = float(np.std(smoothed[-tail_n:]))
    tol       = max(k_std * tail_std, 0.05)
    for i in range(len(smoothed)):
        if np.all(np.abs(smoothed[i:] - final_val) <= tol):
            return i
    return len(mean_returns)


def _bootstrap_ci(seed_means, n_boot=1000, ci=0.95, seed=0):
    """
    Bootstrap confidence interval for the mean of `seed_means`.
    Returns (lo, hi).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(seed_means, dtype=float)
    if arr.size == 0:
        return (float('nan'), float('nan'))
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo_p = (1 - ci) / 2 * 100
    hi_p = (1 + ci) / 2 * 100
    return float(np.percentile(samples, lo_p)), float(np.percentile(samples, hi_p))


def _final_mean_return(result, frac=0.1):
    """Mean return over the last `frac` of episodes, averaged across seeds."""
    n_ep    = result.returns.shape[1]
    cutoff  = max(1, int(n_ep * (1 - frac)))
    return float(result.returns[:, cutoff:].mean())


def _final_std_return(result, frac=0.1):
    """Std of per-seed mean returns in the last `frac` of episodes."""
    n_ep    = result.returns.shape[1]
    cutoff  = max(1, int(n_ep * (1 - frac)))
    return float(result.returns[:, cutoff:].mean(axis=1).std())


# ---------------------------------------------------------------------------
# Summary table (used by multiple plots)
# ---------------------------------------------------------------------------

def compute_summary(results) -> pd.DataFrame:
    """
    Build a tidy summary DataFrame from all experiment results.

    Columns: algorithm, algo_label, setting, horizon, c_switch,
             mean_return, std_return, ci_low, ci_high, sem
    """
    rows = []
    for r in results:
        c       = r.config
        n_ep    = r.returns.shape[1]
        cutoff  = max(1, int(n_ep * 0.9))
        # Per-seed final-mean returns (one value per seed)
        seed_means = r.returns[:, cutoff:].mean(axis=1)
        mean_ret   = float(seed_means.mean())
        std_ret    = float(seed_means.std())
        sem        = std_ret / np.sqrt(max(1, len(seed_means)))
        ci_lo, ci_hi = _bootstrap_ci(seed_means)
        rows.append({
            'algorithm' : c.algorithm,
            'algo_label': ALGO_LABELS[c.algorithm],
            'setting'   : c.setting,
            'horizon'   : c.horizon,
            'c_switch'  : c.c_switch,
            'mean_return': mean_ret,
            'std_return' : std_ret,
            'sem'        : sem,
            'ci_low'     : ci_lo,
            'ci_high'    : ci_hi,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hypothesis analysis functions
# ---------------------------------------------------------------------------

def H1_convergence_by_setting(results) -> pd.DataFrame:
    """
    H1: Higher site separability → faster convergence to optimal policy.

    Returns a DataFrame with convergence_episode for each model-free
    algorithm × setting × horizon × c_switch combination.
    """
    rows = []
    for r in results:
        c = r.config
        if c.algorithm == 'value_iter':
            continue
        mean_curve = r.returns.mean(axis=0)   # (n_episodes,)
        conv_ep    = convergence_episode(mean_curve)
        rows.append({
            'algorithm'           : c.algorithm,
            'algo_label'          : ALGO_LABELS[c.algorithm],
            'setting'             : c.setting,
            'horizon'             : c.horizon,
            'c_switch'            : c.c_switch,
            'convergence_episode' : conv_ep,
            'final_return'        : _final_mean_return(r),
        })
    return pd.DataFrame(rows)


def H2_switching_by_cost(results) -> pd.DataFrame:
    """
    H2: Higher switching cost reduces switching frequency but may increase
    time spent in the non-receptive state.

    Uses converged behavior (last 10 % of episodes).
    """
    rows = []
    for r in results:
        c      = r.config
        if c.algorithm == 'value_iter':
            continue
        n_ep   = r.returns.shape[1]
        cutoff = max(1, int(n_ep * 0.9))

        # switches per episode, normalized by horizon → switches per step
        mean_switch_per_ep = float(r.switch_counts[:, cutoff:].mean())
        switch_rate        = mean_switch_per_ep / c.horizon

        # ps_fracs[:, :, 2] = non-receptive fraction
        nonrec_frac = float(r.ps_fracs[:, cutoff:, 2].mean())
        recep_frac  = float(r.ps_fracs[:, cutoff:, 1].mean())

        rows.append({
            'algorithm'        : c.algorithm,
            'algo_label'       : ALGO_LABELS[c.algorithm],
            'setting'          : c.setting,
            'horizon'          : c.horizon,
            'c_switch'         : c.c_switch,
            'switch_rate'      : switch_rate,
            'mean_switch_count': mean_switch_per_ep,
            'nonreceptive_frac': nonrec_frac,
            'receptive_frac'   : recep_frac,
            'final_return'     : _final_mean_return(r),
        })
    return pd.DataFrame(rows)


def H3_double_q_stability(results) -> pd.DataFrame:
    """
    H3: Double Q-Learning produces more stable learning than vanilla Q-Learning
    (lower variance, less maximization bias).

    Compares return variance and Q-value magnitude across all seeds.
    """
    rows = []
    for r in results:
        c = r.config
        if c.algorithm not in ('qlearning', 'double_q'):
            continue

        # Variance across seeds, averaged over episodes
        return_var = float(r.returns.var(axis=0).mean())

        # Mean absolute Q-value (proxy for maximization bias magnitude)
        q_mag = float(np.abs(r.q_tables[0]).mean()) if r.q_tables else np.nan

        # Per-seed final returns (for box plots)
        n_ep   = r.returns.shape[1]
        cutoff = max(1, int(n_ep * 0.9))
        seed_returns = r.returns[:, cutoff:].mean(axis=1).tolist()  # one value per seed

        rows.append({
            'algorithm'     : c.algorithm,
            'algo_label'    : ALGO_LABELS[c.algorithm],
            'setting'       : c.setting,
            'horizon'       : c.horizon,
            'c_switch'      : c.c_switch,
            'return_variance': return_var,
            'q_magnitude'   : q_mag,
            'mean_return'   : _final_mean_return(r),
            'seed_returns'  : seed_returns,
        })
    return pd.DataFrame(rows)


def H4_model_free_vs_optimal(results) -> pd.DataFrame:
    """
    H4: Value Iteration provides the upper-bound benchmark.

    Returns gap = VI_return − agent_return for every model-free result,
    and gap_pct = gap / |VI_return| × 100.
    """
    # Build VI lookup: (setting, horizon, c_switch) → mean_return
    vi_lookup: dict = {}
    for r in results:
        c = r.config
        if c.algorithm == 'value_iter':
            vi_lookup[(c.setting, c.horizon, c.c_switch)] = _final_mean_return(r)

    rows = []
    for r in results:
        c = r.config
        if c.algorithm == 'value_iter':
            continue
        key      = (c.setting, c.horizon, c.c_switch)
        vi_ret   = vi_lookup.get(key, np.nan)
        agent_ret = _final_mean_return(r)
        gap       = vi_ret - agent_ret
        rows.append({
            'algorithm'   : c.algorithm,
            'algo_label'  : ALGO_LABELS[c.algorithm],
            'setting'     : c.setting,
            'horizon'     : c.horizon,
            'c_switch'    : c.c_switch,
            'agent_return': agent_ret,
            'vi_return'   : vi_ret,
            'gap'         : gap,
            'gap_pct'     : gap / (abs(vi_ret) + 1e-8) * 100,
        })
    return pd.DataFrame(rows)


def H5_state_management(results) -> pd.DataFrame:
    """
    H5: Agents that manage patient-state dynamics will maintain more receptive
    states and avoid non-receptive degradation over training.

    Compares early (first 10 %) vs late (last 10 %) patient-state fractions.
    """
    rows = []
    for r in results:
        c      = r.config
        n_ep   = r.returns.shape[1]
        early  = max(1, int(n_ep * 0.10))
        late   = max(1, int(n_ep * 0.90))

        # ps_fracs: (n_seeds, n_episodes, 3) — [baseline, receptive, non_receptive]
        early_ps = r.ps_fracs[:, :early, :].mean(axis=(0, 1))
        late_ps  = r.ps_fracs[:, late:,  :].mean(axis=(0, 1))

        rows.append({
            'algorithm'             : c.algorithm,
            'algo_label'            : ALGO_LABELS.get(c.algorithm, c.algorithm),
            'setting'               : c.setting,
            'horizon'               : c.horizon,
            'c_switch'              : c.c_switch,
            # early training
            'early_baseline'        : float(early_ps[0]),
            'early_receptive'       : float(early_ps[1]),
            'early_nonreceptive'    : float(early_ps[2]),
            # late (converged) training
            'late_baseline'         : float(late_ps[0]),
            'late_receptive'        : float(late_ps[1]),
            'late_nonreceptive'     : float(late_ps[2]),
            # improvement in receptive fraction
            'receptive_improvement' : float(late_ps[1] - early_ps[1]),
            'nonrec_reduction'      : float(early_ps[2] - late_ps[2]),
            'final_return'          : _final_mean_return(r),
        })
    return pd.DataFrame(rows)
