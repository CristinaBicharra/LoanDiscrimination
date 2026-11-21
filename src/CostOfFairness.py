"""Cost for the lender of reducing an interest rate gap.

Interest is computed over the loan term (Price amortization table):

    monthly rate   m   = (1 + annual rate)^(1/12) - 1
    installment    PMT = L * m / (1 - (1 + m)^-n)
    term interest      = n * PMT - L          (contracts that were repaid)

L is the loan amount and n the number of installments of the contract. The
simulated dataset does not record how many installments a defaulting client
paid before defaulting, so defaulted contracts contribute no interest
(`paid_installments_col=None`). When that column exists, a defaulted
contract earns the interest part of the k installments it paid.

Policy: in every comparable cell where the protected group pays more
(gap > 0) and whose credit score bin lies at or below the cutoff, the
protected group's annual rate is reduced by alpha x gap. The cost is the
interest income the lender gives up, in currency and as a percentage of the
portfolio's term interest income before the change.

`eq4_profit` keeps the paper's Eq. 4 (annual rate x principal for repaid
contracts, minus the principal lost to default) only for comparison: it
ignores the loan term and overstates interest on short loans.
"""

import numpy as np
import pandas as pd

from .DiscriminationMetrics import MIN_N, cell_table

ALPHAS = np.round(np.arange(0, 1.0001, 0.1), 2)
CUTOFFS = {'0.1': 0.1, '0.2': 0.2, '0.4': 0.4, 'all': np.inf}


def monthly_rate(annual_rate_pct):
    """Equivalent monthly rate (fraction) of an annual rate in percent."""
    return (1 + np.asarray(annual_rate_pct, float) / 100) ** (1 / 12) - 1


def price_installment(loan, m, n):
    """Fixed installment of the Price table (zero rate: loan / n)."""
    loan, m, n = (np.asarray(a, float) for a in (loan, m, n))
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.where(m > 0, m / (1 - (1 + m) ** (-n)), 1.0 / n)
    return loan * factor


def term_interest_income(loan, annual_rate_pct, installments, default, paid_installments=None):
    """Interest received over the term of each contract (array)."""
    loan, n, d = (np.asarray(a, float) for a in (loan, installments, default))
    m = monthly_rate(annual_rate_pct)
    pmt = price_installment(loan, m, n)
    repaid = n * pmt - loan
    if paid_installments is None:
        defaulted = np.zeros_like(loan)
    else:
        k = np.minimum(np.asarray(paid_installments, float), n)
        with np.errstate(divide='ignore', invalid='ignore'):
            balance = np.where(m > 0, loan * ((1 + m) ** n - (1 + m) ** k) / ((1 + m) ** n - 1),
                               loan * (1 - k / n))
        defaulted = k * pmt - (loan - balance)
    return (1 - d) * repaid + d * defaulted


def eq4_profit(loan, annual_rate_pct, default):
    """Paper's Eq. 4: sum((1 - d) L r) - sum(d L), with r the annual rate as a fraction."""
    loan, d = np.asarray(loan, float), np.asarray(default, float)
    r = np.asarray(annual_rate_pct, float) / 100
    return float(((1 - d) * loan * r).sum() - (d * loan).sum())


def contract_cell_gap(df, group_col, protected, reference, rate_col, cell_keys, min_n=MIN_N):
    """Gap of each contract's cell (NaN when the cell is not valid)."""
    t = cell_table(df, group_col, protected, reference, rate_col, cell_keys, min_n)
    t = t.loc[t['valid'], list(cell_keys) + ['gap']]
    return df[list(cell_keys)].merge(t, on=list(cell_keys), how='left')['gap'].to_numpy(float)


def cost_curve(df, group_col, protected, reference, rate_col, cell_keys, score_bin_col,
               loan_col, installments_col, default_col, paid_installments_col=None,
               alphas=ALPHAS, cutoffs=CUTOFFS, min_n=MIN_N):
    """Cost of closing a share alpha of the cell gap, for each score cutoff.

    `score_bin_col` holds pandas Intervals; a cell is eligible when the upper
    edge of its score bin is <= the cutoff. Returns one row per (cutoff,
    alpha) with the cost in currency, as % of the term interest income, and
    the Eq. 4 cost for comparison (in currency and as % of the Eq. 4
    interest income, sum over repaid contracts of L x annual rate).
    """
    rate0 = df[rate_col].to_numpy(float)
    gap = contract_cell_gap(df, group_col, protected, reference, rate_col, cell_keys, min_n)
    upper = df[score_bin_col].map(lambda x: x.right if pd.notna(x) else np.nan).astype(float).to_numpy()
    is_prot = (df[group_col] == protected).to_numpy()
    args = (df[loan_col], df[installments_col], df[default_col])
    paid = df[paid_installments_col] if paid_installments_col else None

    def income(rate):
        return float(term_interest_income(args[0], rate, args[1], args[2], paid).sum())

    base_income = income(rate0)
    base_eq4 = eq4_profit(df[loan_col], rate0, df[default_col])
    d = df[default_col].to_numpy(float)
    eq4_income = float(((1 - d) * df[loan_col].to_numpy(float) * rate0 / 100).sum())
    rows = []
    for name, cut in cutoffs.items():
        eligible = is_prot & (upper <= cut + 1e-9) & np.isfinite(gap) & (gap > 0)
        reduction = np.where(eligible, gap, 0.0)
        for a in alphas:
            rate = rate0 - a * reduction
            cost = base_income - income(rate)
            cost4 = base_eq4 - eq4_profit(df[loan_col], rate, df[default_col])
            rows.append({'score_cutoff': name, 'alpha': float(a),
                         'contracts_benefited': int(eligible.sum()) if a > 0 else 0,
                         'cost': cost, 'cost_pct_interest_income': 100 * cost / base_income,
                         'eq4_cost': cost4, 'eq4_cost_pct_eq4_income': 100 * cost4 / eq4_income})
    out = pd.DataFrame(rows)
    out.attrs['interest_income'] = base_income
    out.attrs['eq4_profit'] = base_eq4
    out.attrs['eq4_interest_income'] = eq4_income
    return out
