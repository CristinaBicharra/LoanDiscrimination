"""Runs the paper's income case study on a SYNTHETIC dataset.

    python run_income_example.py             # uses data/Synthetic_Income_Dataset.csv
    python run_income_example.py --regenerate  # rebuilds the dataset first

The bank data are confidential; data/Synthetic_Income_Dataset.csv is fully
artificial (see src/SyntheticIncome.py) and has the columns the method needs.
The script repeats, step by step, what the paper does with the bank data and
writes to outputs/income_example/:

    results_income.md                      report with every number and table
    fig_data_distributions.png             distribution of the main variables by income group
    fig4_interest_rate_histogram.png       number of contracts by annual rate and income group
    fig6_cost_curve.png                    cost of closing a share of the gap, by score cutoff
    fig7_rate_by_score_and_group.png       annual rate by score bin for five attributes
    fig8_default_by_income.png             default rate by score bin and income group
    table4_score_income.csv                Table 4 of the paper (score bins x income group)
    table5_cost_by_score.csv               Table 5 of the paper (cost by score bin)
    cells.csv, cost_of_fairness.csv, wdi_bands.csv

The numbers are NOT findings about any bank: the data were built with a known
income penalty (src/SyntheticIncome.INCOME_PENALTY_PP) to show how the method
recovers it.
"""

import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.CostOfFairness import cost_curve, term_interest_income  # noqa: E402
from src.DiscriminationMetrics import (cell_table, fixed_effects_test,  # noqa: E402
                                       rate_gap_metrics, tost, weighted_disparity_index)
from src.SyntheticIncome import INCOME_PENALTY_PP, make_synthetic_income  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'Synthetic_Income_Dataset.csv')
OUT = os.path.join(ROOT, 'outputs', 'income_example')

# Same credit-score ranges as Table 4 of the paper (plus a first bin for score <= 0.04).
SCORE_BINS = [0, 0.04, 0.06, 0.08, 0.1, 0.14, 0.18, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9]
LTI_BINS = [0, 0.1, 0.2, 0.35, 0.5, 1, 3]                 # loan-to-income ratio
LOAN_BINS = [0, 200, 400, 700, 1000, 2000, 5000, np.inf]  # absolute loan amount (robustness)
RATE_BANDS = [0, 12, 35, 60, 95, 130, 150, 235, 450]      # annual rate bands for the WDI
LOW, HIGH = '< 2 MW', '>= 2 MW'

C_LOW, C_HIGH, C3, C4 = '#eb6834', '#2a78d6', '#1baf7a', '#eda100'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'axes.axisbelow': True,
                     'legend.frameon': False, 'lines.linewidth': 2})
SCORE_LABEL = 'Credit score range (higher = worse credit)'


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches='tight')
    plt.close(fig)


def fmt(x, nd=2):
    return f'{x:,.{nd}f}'


def md_table(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join(lines)


# ---------------------------------------------------------------- figures

def fig_distributions(d):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.6))
    specs = [('MonthlyIncome', 'Monthly income (R$, log scale)', True),
             ('CreditScore', 'Credit score', False),
             ('LoanToIncome', 'Loan-to-income ratio (log scale)', True),
             ('LoanAmount', 'Loan amount (R$, log scale)', True),
             ('InterestRate', 'Annual interest rate (%)', False),
             ('Installments', 'Number of installments', False)]
    for ax, (col, lab, log) in zip(axs.flat, specs):
        x = d[col]
        bins = (np.logspace(np.log10(x.min()), np.log10(x.max()), 30) if log
                else np.linspace(x.min(), x.max(), 30))
        for g, c, ls in ((LOW, C_LOW, '-'), (HIGH, C_HIGH, '--')):
            ax.hist(d.loc[d.IncomeGroup == g, col], bins=bins, histtype='step', color=c,
                    linestyle=ls, linewidth=1.8, label=f'Income {g}')
        if log:
            ax.set_xscale('log')
        ax.set_xlabel(lab)
        ax.set_ylabel('Contracts')
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, 'fig_data_distributions.png')


def by_score(d, mask, col):
    sb = pd.cut(d.CreditScore, SCORE_BINS[1:])
    t = pd.DataFrame({'sb': sb, 'g': mask, 'y': d[col]}).dropna()
    t = t.groupby(['sb', 'g'], observed=True)['y'].agg(['mean', 'size']).unstack('g')
    for g in (True, False):
        t.loc[t[('size', g)] < 30, ('mean', g)] = np.nan
    mid = np.array([(i.left + i.right) / 2 for i in t.index])
    return mid, t


def fig7(d):
    panels = [('(a) Monthly income', d.IncomeGroup == LOW, 'Income < 2 minimum wages', 'Income >= 2 minimum wages'),
              ('(b) Schooling', d.SchoolingCode < 13, 'Less than high school (code < 13)', 'High school or more'),
              ('(c) Age', d.Age >= 60, '60 or older', '18-59'),
              ('(d) Partner', d.Partner == 'No', 'No partner', 'Married or stable union'),
              ('(e) Sex', d.Sex == 'Female', 'Female', 'Male')]
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 8.4), sharex=True, sharey=True)
    for ax, (title, m, l1, l2) in zip(axs.flat, panels):
        mid, t = by_score(d, m, 'InterestRate')
        ax.plot(mid, t[('mean', True)], color=C_LOW, marker='o', label=l1)
        ax.plot(mid, t[('mean', False)], color=C_HIGH, marker='s', linestyle='--', label=l2)
        ax.set_title(title, loc='left', fontsize=9)
        ax.legend(fontsize=7, loc='lower right')
    axs.flat[-1].axis('off')
    for ax in axs[:, 0]:
        ax.set_ylabel('Annual interest rate (%)')
    axs[2, 0].set_xlabel(SCORE_LABEL)
    axs[1, 1].set_xlabel(SCORE_LABEL)
    axs[1, 1].xaxis.set_tick_params(labelbottom=True)
    fig.tight_layout()
    save(fig, 'fig7_rate_by_score_and_group.png')


def fig8(d):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    mid, t = by_score(d.assign(D=100 * d.Default), d.IncomeGroup == LOW, 'D')
    ax.plot(mid, t[('mean', True)], color=C_LOW, marker='o', label='Income < 2 minimum wages')
    ax.plot(mid, t[('mean', False)], color=C_HIGH, marker='s', linestyle='--', label='Income >= 2 minimum wages')
    ax.axvline(0.4, color='#52514e', linewidth=0.8, linestyle=':')
    ax.set_xlabel(SCORE_LABEL)
    ax.set_ylabel('Default rate (%)')
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    save(fig, 'fig8_default_by_income.png')


def fig4(d):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bins = np.linspace(d.InterestRate.min(), d.InterestRate.max(), 35)
    ax.hist(d.loc[d.IncomeGroup == LOW, 'InterestRate'], bins=bins, histtype='step', color=C_LOW,
            linewidth=2, label='Income < 2 minimum wages')
    ax.hist(d.loc[d.IncomeGroup == HIGH, 'InterestRate'], bins=bins, histtype='step', color=C_HIGH,
            linewidth=2, linestyle='--', label='Income >= 2 minimum wages')
    ax.set_xlabel('Annual interest rate (%)')
    ax.set_ylabel('Number of contracts')
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, 'fig4_interest_rate_histogram.png')


def fig6(curve):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    style = {'0.1': (C_HIGH, 'o', '-'), '0.2': (C_LOW, 's', '--'),
             '0.4': (C3, '^', '-.'), 'all': (C4, 'D', ':')}
    names = {'0.1': 'Score <= 0.1', '0.2': 'Score <= 0.2', '0.4': 'Score <= 0.4', 'all': 'All scores'}
    for k, (c, mk, ls) in style.items():
        s = curve[curve.score_cutoff == k]
        ax.plot(100 * s.alpha, s.cost_pct_interest_income, color=c, marker=mk, linestyle=ls, label=names[k])
    ax.set_xlabel('Share of the interest-rate gap closed (%)')
    ax.set_ylabel('Cost (% of interest revenue)')
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    save(fig, 'fig6_cost_curve.png')


# ---------------------------------------------------------------- analysis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regenerate', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.regenerate or not os.path.exists(DATA):
        make_synthetic_income().to_csv(DATA, index=False)
    d = pd.read_csv(DATA)
    d['ScoreBin'] = pd.cut(d.CreditScore, SCORE_BINS)  # scores are >= 0.01
    d['LTIBin'] = pd.cut(d.LoanToIncome, LTI_BINS, include_lowest=True)
    d['LoanBin'] = pd.cut(d.LoanAmount, LOAN_BINS, include_lowest=True)
    g = ('IncomeGroup', LOW, HIGH)
    rep = ['# Income case study on synthetic data', '',
           'Fully artificial data (`data/Synthetic_Income_Dataset.csv`, built by `src/SyntheticIncome.py`). '
           f'By construction, borrowers earning less than R$ 2,000 have the same credit-score and loan-to-income '
           f'distributions as the others but pay {INCOME_PENALTY_PP:.0f} p.p. more per year. '
           'These are not findings about any bank.', '']

    # 1. data
    s = d.groupby('IncomeGroup').agg(
        contracts=('ContractID', 'size'), clients=('ClientID', 'nunique'),
        median_income=('MonthlyIncome', 'median'), mean_score=('CreditScore', 'mean'),
        median_loan_to_income=('LoanToIncome', 'median'), median_loan=('LoanAmount', 'median'),
        mean_annual_rate=('InterestRate', 'mean'), default_rate=('Default', 'mean')).round(3).reset_index()
    rep += ['## 1. The data', '', md_table(s), '',
            'Same score and loan-to-income ratio; different rates. Distributions: `fig_data_distributions.png`.', '']
    fig_distributions(d)

    # 2. Table 4 (score ranges only)
    t4 = d.dropna(subset=['ScoreBin']).groupby(['ScoreBin', 'IncomeGroup'], observed=True).agg(
        default=('Default', 'mean'), qty=('ContractID', 'size'), ir=('InterestRate', 'mean'),
        loan=('LoanAmount', 'mean')).unstack('IncomeGroup')
    tab4 = pd.DataFrame({'Credit score range': t4.index.astype(str),
                         'Default <2MW': t4[('default', LOW)].round(2).values,
                         'Qty <2MW': t4[('qty', LOW)].values,
                         'Avg IR <2MW': t4[('ir', LOW)].round(2).values,
                         'Loan <2MW': t4[('loan', LOW)].round(2).values,
                         'Default >=2MW': t4[('default', HIGH)].round(2).values,
                         'Qty >=2MW': t4[('qty', HIGH)].values,
                         'Avg IR >=2MW': t4[('ir', HIGH)].round(2).values,
                         'Loan >=2MW': t4[('loan', HIGH)].round(2).values})
    tab4['IR difference'] = (tab4['Avg IR <2MW'] - tab4['Avg IR >=2MW']).round(2)
    tab4['IR % difference'] = (100 * tab4['IR difference'] / (100 + tab4['Avg IR >=2MW'])).round(1)
    tab4.to_csv(os.path.join(OUT, 'table4_score_income.csv'), index=False)
    rep += ['## 2. Table 4: annual rate by credit-score range and income group', '', md_table(tab4), '']

    # 3. Eq. 1 and Eq. 2 in score x loan-to-income cells
    cells = cell_table(d, *g, 'InterestRate', ['ScoreBin', 'LTIBin'])
    cells.astype({'ScoreBin': str, 'LTIBin': str}).to_csv(os.path.join(OUT, 'cells.csv'), index=False)
    m = rate_gap_metrics(cells)
    rep += ['## 3. Fairness metrics (Eq. 1 and Eq. 2) in score x loan-to-income cells', '',
            f"- Valid cells (>= 30 contracts in each group): {m['cells_valid']} of {m['cells_total']} (contracts in valid cells: {m['contracts_in_valid_cells']:,})",
            f"- Eq. 1 (simple mean of cell gaps): {fmt(m['eq1'])} p.p.",
            f"- Eq. 2 (weighted by cell size): {fmt(m['eq2'])} p.p.", '']

    # 4. tests
    fe = fixed_effects_test(d, *g, 'InterestRate', ['ScoreBin', 'LTIBin'], cluster_col='ClientID')
    fe_loan = fixed_effects_test(d, *g, 'InterestRate', ['ScoreBin', 'LoanBin', 'Installments'],
                                 cluster_col='ClientID')
    rep += ['## 4. Statistical test: cell fixed effects, standard errors clustered by client', '',
            f"- Score x loan-to-income cells: lower-income borrowers pay {fmt(fe['b'])} p.p. more "
            f"(95% CI {fmt(fe['ci95'][0])} to {fmt(fe['ci95'][1])}; p = {fe['p_value']:.2g}). "
            f"True value built into the data: {INCOME_PENALTY_PP:.0f} p.p.",
            f"- Robustness, score x absolute loan amount x installments: {fmt(fe_loan['b'])} p.p. "
            f"(95% CI {fmt(fe_loan['ci95'][0])} to {fmt(fe_loan['ci95'][1])}). In these synthetic data the price "
            'does not depend on loan size, so the gap does not reverse; in the bank data it does (see the paper).', '',
            'Equivalence test (TOST) on the main estimate:', '',
            md_table(tost(fe['b'], fe['se'], fe['dof']).round(3)), '']

    # 5. WDI
    w = weighted_disparity_index(d, *g, 'InterestRate', RATE_BANDS)
    w['table'].to_csv(os.path.join(OUT, 'wdi_bands.csv'), index=False)
    rep += ['## 5. Weighted Demographic Fairness Index', '',
            f"- With group shares and weights e^-i: {w['wdi_shares_exp']:.3f}",
            f"- With group shares, no weights: {w['wdi_shares_unweighted']:.3f}",
            f"- With absolute counts (Eq. 3 as first written, for comparison): {w['wdi_counts_exp']:.3f}", '']

    # 6. cost of fairness
    curve = cost_curve(d, *g, 'InterestRate', ['ScoreBin', 'LTIBin'], 'ScoreBin', 'LoanAmount',
                       'Installments', 'Default', 'PaidInstallments')
    curve.round(4).to_csv(os.path.join(OUT, 'cost_of_fairness.csv'), index=False)
    full = curve[(curve.score_cutoff == '0.4') & (curve.alpha == 1.0)].iloc[0]
    rep += ['## 6. Cost of fairness', '',
            f"- Interest revenue over each loan's term: R$ {fmt(curve.attrs['interest_income'], 0)}",
            f"- Closing 100% of the gap for lower-income borrowers with score <= 0.4: "
            f"{int(full.contracts_benefited):,} contracts, R$ {fmt(full.cost, 0)}, "
            f"{fmt(full.cost_pct_interest_income)}% of interest revenue "
            f"(Eq. 4 comparison: {fmt(full.eq4_cost_pct_eq4_income)}%).",
            '- The cost grows linearly with the share of the gap closed (`fig6_cost_curve.png`).', '']

    # Table 5: cost by score range (cutoff 0.4, 100%)
    from src.CostOfFairness import contract_cell_gap
    gap = contract_cell_gap(d, *g, 'InterestRate', ['ScoreBin', 'LTIBin'])
    upper = d.ScoreBin.map(lambda x: x.right).astype(float).to_numpy()
    red = np.where((d.IncomeGroup == LOW).to_numpy() & (upper <= 0.4 + 1e-9) & np.isfinite(gap) & (gap > 0), gap, 0)
    base = term_interest_income(d.LoanAmount, d.InterestRate, d.Installments, d.Default, d.PaidInstallments)
    new = term_interest_income(d.LoanAmount, d.InterestRate - red, d.Installments, d.Default, d.PaidInstallments)
    t5 = pd.DataFrame({'sb': d.ScoreBin.astype(str), 'ben': red > 0, 'red': red, 'cost': base - new})
    t5 = t5[t5.ben].groupby('sb', sort=False).agg(contracts=('ben', 'size'), avg_reduction=('red', 'mean'),
                                                     cost=('cost', 'sum')).reset_index()
    t5['cost_pct_interest_revenue'] = 100 * t5.cost / base.sum()
    t5 = t5.rename(columns={'sb': 'Credit score range'}).round(2)
    t5.to_csv(os.path.join(OUT, 'table5_cost_by_score.csv'), index=False)
    rep += ['Table 5: cost by credit-score range (score <= 0.4, 100% of the gap):', '', md_table(t5), '']

    fig4(d)
    fig6(curve)
    fig7(d)
    fig8(d)
    rep += ['## Figures', '', '![](fig_data_distributions.png)', '', '![](fig4_interest_rate_histogram.png)', '',
            '![](fig7_rate_by_score_and_group.png)', '', '![](fig8_default_by_income.png)', '',
            '![](fig6_cost_curve.png)', '']
    with open(os.path.join(OUT, 'results_income.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(rep))
    print('\n'.join(rep[:40]))


if __name__ == '__main__':
    main()
