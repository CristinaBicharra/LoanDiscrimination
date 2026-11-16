"""Cost for the bank of reducing an interest rate disparity.

Following the paper, the expected revenue lost in a credit score bin is

    cost = paid_share * avg_loan * rate_reduction * n_contracts

where `rate_reduction` is expressed as a fraction (an interest rate of
88.20% enters as 0.8820) and `paid_share` is the share of contracts that
did not default. The dataset stores interest rates in percent, so rate
differences are divided by 100 before entering the formula.
"""

import numpy as np
import pandas as pd

PERCENT = 100.0


def _ordered_bins(series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        present = set(series.dropna().unique())
        return [c for c in series.cat.categories if c in present]
    return sorted(series.dropna().unique())


def _bin_cost(data, bin_label, loan_var, credit_score_bin, parameter,
              discriminated, discriminator, price_var, share, default_var):
    in_bin = data[data[credit_score_bin] == bin_label]
    disc = in_bin[in_bin[parameter] == discriminated]
    dom = in_bin[in_bin[parameter] == discriminator]
    if len(disc) == 0 or len(dom) == 0:
        return 0.0
    difference = disc[price_var].mean() - dom[price_var].mean()
    if not difference > 0:
        return 0.0
    paid_share = 1 - disc[default_var].mean() if default_var else 1.0
    return (paid_share * disc[loan_var].mean() * (difference / PERCENT)
            * share * len(disc))


def calculate_fairness_cost(df, Loan_var, CreditScoreBin, parameter, Discriminator,
                            Discriminated, price_discriminator, fairness_degree,
                            max_category, Default_var):
    """Cost of closing a fixed share of the interest rate gap.

    Parameters
    - df: loans, one row per contract.
    - Loan_var: loan amount column.
    - CreditScoreBin: credit score bin column (lowest scores first).
    - parameter: demographic column (e.g. 'Gender').
    - Discriminator / Discriminated: dominant and discriminated groups.
    - price_discriminator: interest rate column, in percent.
    - fairness_degree: share of the gap to close (0 to 1).
    - max_category: number of credit score bins, starting from the best
      scores, that receive the adjustment.
    - Default_var: default column (1 if the client defaulted, 0 otherwise).

    Returns the total cost in currency units.
    """
    labels = _ordered_bins(df[CreditScoreBin])
    total = sum(
        _bin_cost(df, label, Loan_var, CreditScoreBin, parameter, Discriminated,
                  Discriminator, price_discriminator, fairness_degree, Default_var)
        for label in labels[:max_category])
    return round(float(total), 2)


def calculate_fairness_cost_with_decay(data, Loan_var, CreditScoreBin, parameter,
                                       price_discriminator, discriminated, discriminator,
                                       max_discount, min_discount, max_category,
                                       decay_factor, Default_var=None):
    """Cost of closing a share of the gap that decays as credit scores worsen.

    The share closed in bin i (0 = best scores) is
    max(max_discount - decay_factor * i, min_discount).
    If `Default_var` is given, the cost is weighted by the share of paid
    contracts, as in `calculate_fairness_cost`.
    """
    labels = _ordered_bins(data[CreditScoreBin])
    total = 0.0
    for i, label in enumerate(labels[:max_category]):
        share = max(max_discount - decay_factor * i, min_discount)
        total += _bin_cost(data, label, Loan_var, CreditScoreBin, parameter,
                           discriminated, discriminator, price_discriminator,
                           share, Default_var)
    return round(float(total), 2)
