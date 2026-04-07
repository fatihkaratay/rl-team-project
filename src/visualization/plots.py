"""
Phase 6: Visualization

All 8 publication-quality figures for the paper and presentation.
Each function saves figures/<name>.pdf and figures/<name>.png at 300 DPI.

Usage
-----
from src.experiments.runner import run_all
from src.visualization.plots import plot_all

results = run_all()
plot_all(results)
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.metrics import (
    ALGO_LABELS, MODEL_FREE_ALGOS, PATIENT_STATE_LABELS, SITE_LABELS,
    smooth, compute_summary,
    H1_convergence_by_setting, H2_switching_by_cost,
    H3_double_q_stability, H4_model_free_vs_optimal, H5_state_management,
)

# ---------------------------------------------------------------------------
# Paths & style
# ---------------------------------------------------------------------------

_HERE       = os.path.dirname(__file__)
_PROJ_ROOT  = os.path.dirname(os.path.dirname(_HERE))
FIGURES_DIR = os.path.join(_PROJ_ROOT, 'figures')

ALGO_COLORS = {
    'mc':             '#1f77b4',
    'qlearning':      '#ff7f0e',
    'expected_sarsa': '#2ca02c',
    'double_q':       '#d62728',
    'value_iter':     '#9467bd',
}
SETTING_TITLES = {
    'high':     'High Separation',
    'moderate': 'Moderate Separation',
    'low':      'Low Separation',
}
PS_COLORS = ['#4c9be8', '#2ca02c', '#d62728']   # baseline, receptive, non-receptive
SITE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def _setup_style():
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.15)
    plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300})


def _save(fig, name: str):
    """Save figure as PDF + PNG to figures/."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    base = os.path.join(FIGURES_DIR, name)
    fig.savefig(base + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved  figures/{name}.pdf  +  .png')


def _filter(results, **kwargs):
    """Return results matching all key=value filters from config."""
    out = []
    for r in results:
        c = r.config
        if all(getattr(c, k) == v for k, v in kwargs.items()):
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Figure 1 — Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(results, horizon=10, c_switch=0.0,
                         smooth_window=100, **kwargs):
    """
    Episode return vs episodes for all 5 algorithms, one subplot per setting.
    Smoothed mean ± 1 std across seeds.  VI shown as a horizontal dashed line.
    One figure per horizon value (call twice for horizon=5 and horizon=10).
    """
    _setup_style()
    settings = ['high', 'moderate', 'low']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle(
        f'Learning Curves  |  horizon={horizon}, $c_{{switch}}$={c_switch}',
        fontsize=13, y=1.02,
    )

    for ax, setting in zip(axes, settings):
        subset = _filter(results, horizon=horizon, c_switch=c_switch,
                         setting=setting)

        # Horizontal reference line for Value Iteration
        vi_results = [r for r in subset if r.config.algorithm == 'value_iter']
        if vi_results:
            vi_ret = vi_results[0].returns.mean()
            ax.axhline(vi_ret, color=ALGO_COLORS['value_iter'],
                       linestyle='--', linewidth=1.5,
                       label=ALGO_LABELS['value_iter'])

        for algo in MODEL_FREE_ALGOS:
            algo_res = [r for r in subset if r.config.algorithm == algo]
            if not algo_res:
                continue
            r = algo_res[0]
            mean_curve = r.returns.mean(axis=0)
            std_curve  = r.returns.std(axis=0)
            sm_mean    = smooth(mean_curve, smooth_window)
            sm_std     = smooth(std_curve,  smooth_window)
            eps        = np.arange(len(sm_mean))
            color      = ALGO_COLORS[algo]
            ax.plot(eps, sm_mean, color=color,
                    linewidth=1.8, label=ALGO_LABELS[algo])
            ax.fill_between(eps,
                            sm_mean - sm_std,
                            sm_mean + sm_std,
                            color=color, alpha=0.15)

        ax.set_title(SETTING_TITLES[setting], fontsize=11)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Return' if setting == 'high' else '')
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))

    # Shared legend below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    # Collect from all axes in case some algos only appear in some subplots
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    fig.legend(handles, labels, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.08), frameon=True, fontsize=9)

    fig.tight_layout()
    _save(fig, f'fig1_learning_curves_h{horizon}_c{c_switch}')
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Performance heatmap
# ---------------------------------------------------------------------------

def plot_performance_heatmap(results, horizon=10, c_switch=0.0, **kwargs):
    """
    Algorithm × Setting → mean final return.  A 5 × 3 annotated heatmap.
    """
    _setup_style()
    df  = compute_summary(results)
    sub = df[(df.horizon == horizon) & (df.c_switch == c_switch)]

    algo_order    = ['value_iter', 'mc', 'qlearning', 'expected_sarsa', 'double_q']
    setting_order = ['high', 'moderate', 'low']

    pivot = (sub
             .pivot_table(index='algorithm', columns='setting',
                          values='mean_return', aggfunc='mean')
             .reindex(index=algo_order, columns=setting_order))

    # Build annotation strings  "mean ± std"
    pivot_std = (sub
                 .pivot_table(index='algorithm', columns='setting',
                              values='std_return', aggfunc='mean')
                 .reindex(index=algo_order, columns=setting_order))
    annot = pivot.round(3).astype(str) + '\n±' + pivot_std.round(3).astype(str)

    row_labels = [ALGO_LABELS[a] for a in algo_order]
    col_labels = [SETTING_TITLES[s] for s in setting_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot, annot=annot, fmt='', cmap='RdYlGn',
        xticklabels=col_labels, yticklabels=row_labels,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Mean Final Return'},
        ax=ax,
    )
    ax.set_title(
        f'Final Performance Heatmap  |  horizon={horizon}, $c_{{switch}}$={c_switch}',
        fontsize=12, pad=10,
    )
    ax.set_xlabel('Setting'); ax.set_ylabel('')
    fig.tight_layout()
    _save(fig, f'fig2_performance_heatmap_h{horizon}_c{c_switch}')
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Patient state distribution (late training)
# ---------------------------------------------------------------------------

def plot_patient_state_distribution(results, horizon=10, c_switch=0.0, **kwargs):
    """
    Three-panel grouped bar chart: one subplot per patient state (Baseline /
    Receptive / Non-Receptive), x-axis = setting, bars = algorithms.
    Uses converged behaviour (last 10 % of training episodes).
    """
    _setup_style()
    df  = H5_state_management(results)
    sub = df[(df.horizon == horizon) & (df.c_switch == c_switch)]

    settings   = ['high', 'moderate', 'low']
    algos      = MODEL_FREE_ALGOS + ['value_iter']
    ps_cols    = ['late_baseline', 'late_receptive', 'late_nonreceptive']
    ps_titles  = ['Baseline', 'Receptive', 'Non-Receptive']

    n_algos = len(algos)
    width   = 0.7 / n_algos
    x_base  = np.arange(len(settings))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle(
        f'Patient State Distribution (Converged)  |  '
        f'horizon={horizon}, $c_{{switch}}$={c_switch}',
        fontsize=12, y=1.02,
    )

    for ax, ps_col, ps_title in zip(axes, ps_cols, ps_titles):
        for i, algo in enumerate(algos):
            offset    = (i - n_algos / 2 + 0.5) * width
            algo_data = sub[sub.algorithm == algo]
            vals = []
            for s in settings:
                row = algo_data[algo_data.setting == s]
                vals.append(float(row[ps_col].values[0]) if not row.empty else 0.0)
            ax.bar(x_base + offset, vals, width,
                   color=ALGO_COLORS[algo], label=ALGO_LABELS[algo], alpha=0.85)

        ax.set_title(ps_title, fontsize=11)
        ax.set_xticks(x_base)
        ax.set_xticklabels([SETTING_TITLES[s] for s in settings], rotation=12, ha='right')
        ax.set_ylabel('Fraction of Episode Steps' if ps_title == 'Baseline' else '')
        ax.set_ylim(0, 0.75)

    # Single shared legend below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=n_algos,
               bbox_to_anchor=(0.5, -0.08), frameon=True, fontsize=9)

    fig.tight_layout()
    _save(fig, f'fig3_patient_state_dist_h{horizon}_c{c_switch}')
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Site visit frequency
# ---------------------------------------------------------------------------

def plot_site_visit_frequency(results, setting='high', horizon=10,
                              c_switch=0.0, **kwargs):
    """
    Grouped bar chart: fraction of steps at each site (S1–S4) per algorithm,
    using converged behavior (last 10 % of training episodes).
    """
    _setup_style()
    subset = _filter(results, setting=setting, horizon=horizon, c_switch=c_switch)
    algos  = MODEL_FREE_ALGOS + ['value_iter']

    n_algos = len(algos)
    n_sites = 4
    width   = 0.7 / n_sites
    x_base  = np.arange(n_algos)

    fig, ax = plt.subplots(figsize=(11, 4.5))

    for j, site_idx in enumerate(range(n_sites)):
        offset = (j - n_sites / 2 + 0.5) * width
        fracs  = []
        for algo in algos:
            algo_res = [r for r in subset if r.config.algorithm == algo]
            if not algo_res:
                fracs.append(0.0)
                continue
            r      = algo_res[0]
            n_ep   = r.site_visit_fracs.shape[1]
            cutoff = max(1, int(n_ep * 0.9))
            # site_visit_fracs: (n_seeds, n_episodes, 4)
            frac = float(r.site_visit_fracs[:, cutoff:, site_idx].mean())
            fracs.append(frac)
        ax.bar(x_base + offset, fracs, width,
               color=SITE_COLORS[site_idx], label=SITE_LABELS[site_idx])

    ax.set_xticks(x_base)
    ax.set_xticklabels([ALGO_LABELS[a] for a in algos], rotation=15, ha='right')
    ax.set_ylabel('Fraction of Steps')
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f'Site Visit Frequency (Converged)  |  {SETTING_TITLES[setting]}, '
        f'horizon={horizon}, $c_{{switch}}$={c_switch}',
        fontsize=11,
    )
    ax.legend(title='Site', ncol=4, fontsize=9)
    fig.tight_layout()
    _save(fig, f'fig4_site_freq_{setting}_h{horizon}_c{c_switch}')
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Switching frequency vs switch cost
# ---------------------------------------------------------------------------

def plot_switching_vs_cost(results, setting='high', horizon=10, **kwargs):
    """
    Line plot: switch rate (per step) vs c_switch, one line per model-free algorithm.
    """
    _setup_style()
    df  = H2_switching_by_cost(results)
    sub = df[(df.setting == setting) & (df.horizon == horizon)]

    costs = sorted(sub.c_switch.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: switching rate
    for algo in MODEL_FREE_ALGOS:
        algo_data = sub[sub.algorithm == algo]
        rates     = [float(algo_data[algo_data.c_switch == c].switch_rate.mean())
                     for c in costs]
        axes[0].plot(costs, rates, marker='o', color=ALGO_COLORS[algo],
                     linewidth=2, markersize=6, label=ALGO_LABELS[algo])
    axes[0].set_xlabel('Switching Cost $c_{switch}$')
    axes[0].set_ylabel('Switch Rate (switches / step)')
    axes[0].set_title('Switching Frequency vs Cost')
    axes[0].set_xticks(costs)
    axes[0].legend(fontsize=9)

    # Right: non-receptive fraction
    for algo in MODEL_FREE_ALGOS:
        algo_data = sub[sub.algorithm == algo]
        nrfrac    = [float(algo_data[algo_data.c_switch == c].nonreceptive_frac.mean())
                     for c in costs]
        axes[1].plot(costs, nrfrac, marker='s', color=ALGO_COLORS[algo],
                     linewidth=2, markersize=6, label=ALGO_LABELS[algo])
    axes[1].set_xlabel('Switching Cost $c_{switch}$')
    axes[1].set_ylabel('Fraction of Steps (Non-Receptive)')
    axes[1].set_title('Non-Receptive State Fraction vs Cost')
    axes[1].set_xticks(costs)
    axes[1].legend(fontsize=9)

    fig.suptitle(
        f'H2: Switching Cost Effects  |  {SETTING_TITLES[setting]}, horizon={horizon}',
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, f'fig5_switching_vs_cost_{setting}_h{horizon}')
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Q-value heatmap (Value Iteration)
# ---------------------------------------------------------------------------

def plot_q_value_heatmap(results, setting='high', horizon=10,
                         c_switch=0.0, **kwargs):
    """
    Heatmap of Q*(s, a) from Value Iteration at t=0.
    Rows: (site × patient_state) combinations for sites S1–S4.
    Columns: actions S1–S4.
    """
    _setup_style()
    vi_res = _filter(results, algorithm='value_iter', setting=setting,
                     horizon=horizon, c_switch=c_switch)
    if not vi_res:
        print('  [skip] no Value Iteration result found for this config')
        return None

    # Q-table shape: (5, 3, horizon+1, 4) — site_idx, ps_idx, t, action
    Q   = vi_res[0].q_tables[0]
    # Take t=0, sites 1-4 (exclude Start=0)
    Q_t0 = Q[1:, :, 0, :]   # shape (4, 3, 4)
    Q_flat = Q_t0.reshape(12, 4)  # 12 state rows, 4 action cols

    row_labels = [
        f'{site}/{ps}'
        for site in SITE_LABELS
        for ps in ['base', 'recep', 'nonrec']
    ]
    col_labels = [f'Act {s}' for s in SITE_LABELS]

    fig, ax = plt.subplots(figsize=(7, 8))
    sns.heatmap(
        Q_flat, annot=True, fmt='.2f', cmap='RdYlGn',
        xticklabels=col_labels, yticklabels=row_labels,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Q*(s, a)'},
        ax=ax,
    )
    ax.set_title(
        f'Optimal Q-Values (VI)  |  {SETTING_TITLES[setting]}, '
        f'horizon={horizon}, t=0',
        fontsize=11, pad=10,
    )
    ax.set_xlabel('Action'); ax.set_ylabel('State (site / patient state)')
    fig.tight_layout()
    _save(fig, f'fig6_qvalue_heatmap_{setting}_h{horizon}')
    return fig


# ---------------------------------------------------------------------------
# Figure 7 — Algorithm gap to optimal (VI upper bound)
# ---------------------------------------------------------------------------

def plot_algorithm_gap(results, horizon=10, c_switch=0.0, **kwargs):
    """
    Bar chart of gap = VI_return − agent_return for each model-free algorithm,
    grouped by setting.  Illustrates H4.
    """
    _setup_style()
    df  = H4_model_free_vs_optimal(results)
    sub = df[(df.horizon == horizon) & (df.c_switch == c_switch)]

    settings = ['high', 'moderate', 'low']
    algos    = MODEL_FREE_ALGOS
    n_algos  = len(algos)
    width    = 0.7 / n_algos
    x_base   = np.arange(len(settings))

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, algo in enumerate(algos):
        offset    = (i - n_algos / 2 + 0.5) * width
        algo_data = sub[sub.algorithm == algo]
        gaps      = [float(algo_data[algo_data.setting == s].gap.mean())
                     for s in settings]
        ax.bar(x_base + offset, gaps, width,
               color=ALGO_COLORS[algo], label=ALGO_LABELS[algo])

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x_base)
    ax.set_xticklabels([SETTING_TITLES[s] for s in settings])
    ax.set_ylabel('Gap to Optimal (VI return − agent return)')
    ax.set_title(
        f'H4: Model-Free vs Optimal  |  horizon={horizon}, $c_{{switch}}$={c_switch}',
        fontsize=12,
    )
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    _save(fig, f'fig7_algorithm_gap_h{horizon}_c{c_switch}')
    return fig


# ---------------------------------------------------------------------------
# Figure 8 — H3 variance: Q-Learning vs Double Q-Learning
# ---------------------------------------------------------------------------

def plot_h3_variance(results, setting='high', horizon=10, c_switch=0.0, **kwargs):
    """
    Box plots of per-seed final returns comparing Q-Learning and Double Q-Learning.
    One panel per setting (or fix to a single setting via the `setting` kwarg).
    Illustrates H3: Double Q-Learning should show tighter distributions.
    """
    _setup_style()
    df  = H3_double_q_stability(results)
    sub = df[(df.horizon == horizon) & (df.c_switch == c_switch)]

    settings = ['high', 'moderate', 'low']
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    fig.suptitle(
        f'H3: Return Stability — Q-Learning vs Double Q-Learning  |  '
        f'horizon={horizon}, $c_{{switch}}$={c_switch}',
        fontsize=12, y=1.02,
    )

    for ax, sett in zip(axes, settings):
        box_data  = []
        box_labels = []
        box_colors = []
        for algo in ('qlearning', 'double_q'):
            row = sub[(sub.algorithm == algo) & (sub.setting == sett)]
            if row.empty:
                continue
            # seed_returns is stored as a list in the DataFrame cell
            seed_rets = row.seed_returns.values[0]
            box_data.append(seed_rets)
            box_labels.append(ALGO_LABELS[algo])
            box_colors.append(ALGO_COLORS[algo])

        if not box_data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(box_data, patch_artist=True, widths=0.4,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(box_labels, fontsize=9)
        ax.set_title(SETTING_TITLES[sett], fontsize=11)
        ax.set_ylabel('Mean Final Return' if sett == 'high' else '')
        ax.set_xlabel('')

    fig.tight_layout()
    _save(fig, f'fig8_h3_variance_h{horizon}_c{c_switch}')
    return fig


# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------

def plot_all(results, **kwargs):
    """
    Generate and save all 8 figures.  Returns list of figure objects.

    Default parameters: horizon=10, c_switch=0.0, setting='high'.
    Override via kwargs, e.g. plot_all(results, horizon=5).
    """
    horizon  = kwargs.pop('horizon',  10)
    c_switch = kwargs.pop('c_switch', 0.0)
    setting  = kwargs.pop('setting',  'high')

    print('Generating figures...')
    figs = []

    # Fig 1: learning curves for both horizons
    for h in (5, 10):
        figs.append(plot_learning_curves(results, horizon=h,
                                         c_switch=c_switch, **kwargs))

    figs.append(plot_performance_heatmap(results, horizon=horizon,
                                         c_switch=c_switch, **kwargs))
    figs.append(plot_patient_state_distribution(results, horizon=horizon,
                                                c_switch=c_switch, **kwargs))
    # Fig 4 & Fig 5: one per separation setting
    for s in ('high', 'moderate', 'low'):
        figs.append(plot_site_visit_frequency(results, setting=s,
                                              horizon=horizon, c_switch=c_switch,
                                              **kwargs))
        figs.append(plot_switching_vs_cost(results, setting=s,
                                           horizon=horizon, **kwargs))
    figs.append(plot_q_value_heatmap(results, setting=setting,
                                     horizon=horizon, c_switch=c_switch,
                                     **kwargs))
    figs.append(plot_algorithm_gap(results, horizon=horizon,
                                   c_switch=c_switch, **kwargs))
    figs.append(plot_h3_variance(results, setting=setting,
                                 horizon=horizon, c_switch=c_switch, **kwargs))

    print(f'Done. {len(figs)} figures saved to {FIGURES_DIR}/')
    return figs
