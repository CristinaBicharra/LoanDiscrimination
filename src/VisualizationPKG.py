"""Plots used in the analysis. Every plot is saved as a PNG file."""

import os

import matplotlib.pyplot as plt
import numpy as np

SCORE_LABEL = 'Credit score (higher = worse credit)'
RATE_LABEL = 'Annual interest rate (%)'
COLORS = {'Male': '#1f77b4', 'Female': '#d62728'}

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
                     'legend.fontsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11})


def _save(fig, output_dir, file_name, show=False):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, file_name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_rate_histogram(df, rate_col, group_col, groups, output_dir='.',
                        file_name='fig_interest_rate_histogram.png', bins=40):
    """Number of contracts by interest rate, one color per group (Fig. 4)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    edges = np.histogram_bin_edges(df[rate_col].dropna(), bins=bins)
    for g in groups:
        ax.hist(df.loc[df[group_col] == g, rate_col], bins=edges, alpha=0.55,
                label=g, color=COLORS.get(g))
    ax.set_xlabel(RATE_LABEL)
    ax.set_ylabel('Number of contracts')
    ax.set_title('Interest rate distribution by gender')
    ax.legend()
    return _save(fig, output_dir, file_name)


def plot_rate_by_score(df, score_col, rate_col, group_col, groups, output_dir='.',
                       file_name='fig_interest_rate_by_score.png'):
    """Interest rate against credit score, one color per group."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for g in groups:
        part = df[df[group_col] == g]
        ax.scatter(part[score_col], part[rate_col], s=8, alpha=0.4, label=g, color=COLORS.get(g))
    ax.set_xlabel(SCORE_LABEL)
    ax.set_ylabel(RATE_LABEL)
    ax.set_title('Interest rate by credit score')
    ax.legend(markerscale=3)
    return _save(fig, output_dir, file_name)


def plot_gap_by_score(table, output_dir='.', file_name='fig_disparity_by_score_bin.png'):
    """Weighted within-cell gap (Eq. 2) for each credit score bin, with 95% interval.

    `table` has columns 'score_bin', 'eq2', 'ci_low', 'ci_high'.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(table))
    err = np.vstack([table['eq2'] - table['ci_low'], table['ci_high'] - table['eq2']])
    ax.bar(x, table['eq2'], color='#8c2d04', width=0.6)
    ax.errorbar(x, table['eq2'], yerr=err, fmt='none', ecolor='black', capsize=4)
    ax.axhline(0, color='grey', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(table['score_bin'].astype(str), rotation=45, ha='right')
    ax.set_xlabel(SCORE_LABEL)
    ax.set_ylabel('Rate gap, women minus men (p.p.)')
    ax.set_title('Interest rate gap within comparable cells, by credit score bin')
    return _save(fig, output_dir, file_name)


def plot_cost_curve(curve, output_dir='.', file_name='fig_cost_of_fairness.png'):
    """Cost (% of term interest income) against the share of the gap closed."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, part in curve.groupby('score_cutoff', sort=False):
        label = 'all scores' if name == 'all' else f'score <= {name}'
        ax.plot(100 * part['alpha'], part['cost_pct_interest_income'], marker='o', label=label)
    ax.set_xlabel('Share of the within-cell rate gap closed (%)')
    ax.set_ylabel('Cost (% of term interest income)')
    ax.set_title('Cost of fairness: lowering women\'s rates toward men\'s')
    ax.legend(title='Contracts adjusted')
    ax.grid(alpha=0.3)
    return _save(fig, output_dir, file_name)
