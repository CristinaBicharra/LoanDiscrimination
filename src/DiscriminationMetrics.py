"""Fairness metrics for loan pricing.

All functions compare a potentially discriminated group (e.g. 'Female')
against a dominant group (e.g. 'Male') on a price variable (the annual
interest rate, in percent).

- calculate_weighted_disparity_index: Weighted Demographic Fairness Index (WDI).
- calculate_equality_of_opportunity: interest rate disparity between groups
  with the same credit score (simple and weighted averages across bins).
- calculate_equality_of_odds_opportunity: the same disparity, split between
  borrowers who repaid and borrowers who defaulted.
- interest_rate_tests_by_bin: Welch t-test of the disparity in each bin.
"""

import numpy as np
import pandas as pd


def _ordered_bins(series):
    """Bins present in the data, in credit score order."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        present = set(series.dropna().unique())
        return [c for c in series.cat.categories if c in present]
    return sorted(series.dropna().unique())


def calculate_weighted_disparity_index(bin_summary, discriminated, discriminator):
    """Weighted Demographic Fairness Index (WDI).

    `bin_summary` has one row per interest rate bin (lowest rates first) and
    one column per group, holding the number of clients.

        WDI = sum_i [ (C_dom,i - C_disc,i) / (C_dom,i + C_disc,i) * w_i ] / sum_i w_i
        w_i = exp(-i)

    Bins with lower interest rates receive higher weights. The index ranges
    from -1 to 1: 1 is maximal disparity favoring the dominant group, 0 is
    no disparity, and -1 is disparity favoring the discriminated group.
    """
    weights = np.exp(-np.arange(len(bin_summary)))
    weighted = []
    for i in range(len(bin_summary)):
        dominant = bin_summary.iloc[i][discriminator]
        disc = bin_summary.iloc[i][discriminated]
        total = dominant + disc
        ratio = (dominant - disc) / total if total > 0 else 0.0
        weighted.append(ratio * weights[i])
    return float(np.sum(weighted) / np.sum(weights))


def calculate_equality_of_opportunity(df, parameter, discriminated, discriminator,
                                      price_discriminator, credit_score_bins):
    """Interest rate disparity under the equality of opportunity criterion.

    For each credit score bin, the disparity is the average rate of the
    discriminated group minus the average rate of the dominant group.

    Returns a dict with:
    - 'simple': average of the bin disparities (Eq. 1 of the paper);
    - 'weighted': average weighted by the number of clients in each bin
      (Eq. 2 of the paper);
    - 'table': DataFrame with one row per bin.

    A positive value means the discriminated group pays more for the same
    credit score; zero means no disparity.
    """
    rows = []
    for credit_bin in _ordered_bins(df[credit_score_bins]):
        df_bin = df[df[credit_score_bins] == credit_bin]
        disc = df_bin.loc[df_bin[parameter] == discriminated, price_discriminator]
        dom = df_bin.loc[df_bin[parameter] == discriminator, price_discriminator]
        if len(disc) == 0 or len(dom) == 0:
            continue
        rows.append({
            'bin': credit_bin,
            'n_' + str(discriminated): len(disc),
            'n_' + str(discriminator): len(dom),
            'rate_' + str(discriminated): disc.mean(),
            'rate_' + str(discriminator): dom.mean(),
            'disparity': disc.mean() - dom.mean(),
            'n': len(df_bin),
        })
    table = pd.DataFrame(rows)
    simple = float(table['disparity'].mean())
    weighted = float((table['disparity'] * table['n']).sum() / table['n'].sum())
    return {'simple': simple, 'weighted': weighted, 'table': table}


def calculate_equality_of_odds_opportunity(df, parameter, discriminated, discriminator,
                                           price_discriminator, cs_bin_var,
                                           par_default, default_threshold=0.5):
    """Disparity split by repayment outcome.

    Returns (eop, eoo, table):
    - eop: weighted disparity among borrowers who did not default
      (equality of opportunity);
    - eoo: eop plus the weighted disparity among borrowers who defaulted
      (equality of odds);
    - table: contribution of each credit score bin to both values.
    """
    total = len(df)
    rows = []
    for label in _ordered_bins(df[cs_bin_var]):
        in_bin = df[df[cs_bin_var] == label]
        contrib = {}
        for name, part in (('Non-Default', in_bin[in_bin[par_default] <= default_threshold]),
                           ('Default', in_bin[in_bin[par_default] > default_threshold])):
            disc = part.loc[part[parameter] == discriminated, price_discriminator]
            dom = part.loc[part[parameter] == discriminator, price_discriminator]
            if len(disc) == 0 or len(dom) == 0:
                contrib[name] = 0.0
            else:
                contrib[name] = (disc.mean() - dom.mean()) * len(part) / total
        rows.append({'Credit Score Bins': label, **contrib})
    table = pd.DataFrame(rows)
    eop = float(table['Non-Default'].sum())
    eoo = float(table['Non-Default'].sum() + table['Default'].sum())
    return eop, eoo, table


def interest_rate_tests_by_bin(df, parameter, discriminated, discriminator,
                               price_discriminator, credit_score_bins):
    """Welch t-test of the interest rate difference in each credit score bin.

    Null hypothesis: within a bin, both groups are charged the same average
    rate. A p-value below 0.05 indicates a statistically significant
    difference between the groups.
    """
    from scipy import stats
    rows = []
    for credit_bin in _ordered_bins(df[credit_score_bins]):
        df_bin = df[df[credit_score_bins] == credit_bin]
        disc = df_bin.loc[df_bin[parameter] == discriminated, price_discriminator]
        dom = df_bin.loc[df_bin[parameter] == discriminator, price_discriminator]
        if len(disc) < 2 or len(dom) < 2:
            continue
        t_stat, p_value = stats.ttest_ind(disc, dom, equal_var=False)
        rows.append({'bin': credit_bin, 't_stat': t_stat, 'p_value': p_value})
    return pd.DataFrame(rows)
