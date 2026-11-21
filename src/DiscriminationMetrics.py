"""Fairness metrics for loan pricing.

Convention: the *protected* group (potentially discriminated, e.g. 'Female')
is compared with the *reference* group (e.g. 'Male') on the annual interest
rate, in percent. A positive gap means the protected group pays more.

Comparisons are made inside *comparable cells*: groups of contracts with the
same credit score bin and the same loan terms. In the paper the cells are
credit score x (loan amount / income); the simulated dataset has no income,
so here they are credit score x loan amount x number of installments.

- cell_table: mean rate of each group in each cell and the cell gap.
- rate_gap_metrics: Eq. 1 (simple mean of the cell gaps) and Eq. 2 (mean
  weighted by the number of contracts in the cell), with the number of cells.
- fixed_effects_test: regression of the rate on a protected-group dummy with
  cell fixed effects and robust standard errors (clustered by client when a
  client id exists, HC1 otherwise). This is the main significance test.
- tost: equivalence test (two one-sided tests). Equality of rates can only be
  claimed through this test, never from a non-significant difference.
- weighted_disparity_index: WDI (Eq. 3), computed on the share of each group
  in each interest rate band, with exponential weights and without weights.
"""

import numpy as np
import pandas as pd
from scipy import stats

MIN_N = 30                       # minimum contracts of EACH group for a valid cell
TOST_MARGINS = (1.0, 2.0, 5.0)   # equivalence margins, percentage points


def _valid_rows(df, group_col, protected, reference, cols):
    d = df.dropna(subset=list(cols) + [group_col])
    return d[d[group_col].isin([protected, reference])]


def cell_table(df, group_col, protected, reference, rate_col, cell_keys, min_n=MIN_N):
    """One row per comparable cell.

    Columns: the cell keys, n/mean/variance of the rate for the protected
    group (n1, m1, v1) and the reference group (n2, m2, v2), `gap` = m1 - m2
    in percentage points, and `valid` (at least `min_n` contracts in each
    group). Cells without both groups have NaN gap and are not valid.
    """
    d = _valid_rows(df, group_col, protected, reference, list(cell_keys) + [rate_col])
    d = d.assign(_g1=d[group_col] == protected)
    agg = d.groupby(list(cell_keys) + ['_g1'], observed=True)[rate_col].agg(['size', 'mean', 'var'])
    agg = agg.unstack('_g1')
    t = pd.DataFrame({
        'n1': agg.get(('size', True)), 'n2': agg.get(('size', False)),
        'm1': agg.get(('mean', True)), 'm2': agg.get(('mean', False)),
        'v1': agg.get(('var', True)), 'v2': agg.get(('var', False)),
    }).reset_index()
    t[['n1', 'n2']] = t[['n1', 'n2']].fillna(0).astype(int)
    t['gap'] = t['m1'] - t['m2']
    t['valid'] = (t['n1'] >= min_n) & (t['n2'] >= min_n)
    return t


def rate_gap_metrics(table):
    """Eq. 1 and Eq. 2 from a `cell_table`, using only valid cells.

    - eq1: simple mean of the cell gaps;
    - eq2: mean of the cell gaps weighted by the number of contracts in the
      cell (n1 + n2).
    Both come with an analytic 95% interval that treats contracts as
    independent (Var = sum w^2 (s1^2/n1 + s2^2/n2)); the formal test is
    `fixed_effects_test`.
    """
    v = table[table['valid']]
    k = len(v)
    out = {'cells_total': int(len(table)), 'cells_valid': int(k),
           'cells_dropped': int(len(table) - k),
           'contracts_total': int(table['n1'].sum() + table['n2'].sum()),
           'contracts_in_valid_cells': int(v['n1'].sum() + v['n2'].sum()),
           'cells_protected_pays_more': int((v['gap'] > 0).sum())}
    if k == 0:
        return out | {'eq1': np.nan, 'eq2': np.nan}
    se2 = v['v1'] / v['n1'] + v['v2'] / v['n2']
    eq1 = float(v['gap'].mean())
    se1 = float(np.sqrt(se2.sum()) / k)
    w = (v['n1'] + v['n2']) / (v['n1'] + v['n2']).sum()
    eq2 = float((w * v['gap']).sum())
    se_2 = float(np.sqrt((w ** 2 * se2).sum()))
    return out | {'eq1': eq1, 'eq1_ci95': (eq1 - 1.96 * se1, eq1 + 1.96 * se1),
                  'eq2': eq2, 'eq2_ci95': (eq2 - 1.96 * se_2, eq2 + 1.96 * se_2)}


def _within_ols(y, X, cell, cluster=None):
    """OLS after removing each cell's mean (cell fixed effects).

    Standard errors: clustered by `cluster` (CR1) if given, else HC1.
    """
    def demean(a):
        s = np.zeros((cell.max() + 1, a.shape[1]))
        np.add.at(s, cell, a)
        return a - (s / np.bincount(cell)[:, None])[cell]
    yd = demean(y[:, None])[:, 0]
    Xd = demean(X)
    bread = np.linalg.pinv(Xd.T @ Xd)
    b = bread @ Xd.T @ yd
    u = yd - Xd @ b
    n, n_cells = len(y), len(np.unique(cell))
    k = Xd.shape[1] + n_cells
    if cluster is not None:
        g = cluster.max() + 1
        scores = np.zeros((g, Xd.shape[1]))
        np.add.at(scores, cluster, Xd * u[:, None])
        n_cl = len(np.unique(cluster))
        c = n_cl / (n_cl - 1) * (n - 1) / max(n - k, 1)
        meat, dof = scores.T @ scores, n_cl - 1
    else:
        c = n / max(n - k, 1)
        meat, dof = (Xd * u[:, None] ** 2).T @ Xd, n - k
    V = c * bread @ meat @ bread
    return b, np.sqrt(np.diag(V)), n, n_cells, dof


def fixed_effects_test(df, group_col, protected, reference, rate_col, cell_keys,
                       cluster_col=None, controls=(), min_n=MIN_N):
    """rate = b * protected + cell fixed effect (+ controls) + error.

    Only contracts in valid cells (see `cell_table`) enter. `b` is the
    average rate difference (p.p.) between the groups within the same cell.
    Returns b, its robust standard error, 95% interval, t and p-value,
    and the number of contracts and cells.
    """
    t = cell_table(df, group_col, protected, reference, rate_col, cell_keys, min_n)
    valid = t.loc[t['valid'], list(cell_keys)]
    d = _valid_rows(df, group_col, protected, reference,
                    list(cell_keys) + [rate_col] + list(controls))
    d = d.merge(valid, on=list(cell_keys), how='inner')
    cell = d.groupby(list(cell_keys), observed=True).ngroup().to_numpy()
    X = np.column_stack([(d[group_col] == protected).astype(float).to_numpy()]
                        + [d[c].astype(float).to_numpy() for c in controls])
    cl = pd.factorize(d[cluster_col])[0] if cluster_col else None
    b, se, n, n_cells, dof = _within_ols(d[rate_col].to_numpy(float), X, cell, cl)
    q = stats.t.ppf(0.975, dof)
    tval = b[0] / se[0]
    return {'b': float(b[0]), 'se': float(se[0]), 'ci95': (b[0] - q * se[0], b[0] + q * se[0]),
            't': float(tval), 'p_value': float(2 * stats.t.sf(abs(tval), dof)), 'dof': int(dof),
            'n': int(n), 'cells': int(n_cells),
            'se_type': f'clustered by {cluster_col}' if cluster_col else 'HC1'}


def tost(b, se, dof, margins=TOST_MARGINS):
    """Two one-sided tests of equivalence: is |true gap| < margin?

    Equivalent at the 5% level when the 90% interval lies inside
    (-margin, +margin). Returns one row per margin.
    """
    q = stats.t.ppf(0.95, dof)
    lo, hi = b - q * se, b + q * se
    rows = []
    for m in margins:
        p = max(stats.t.sf((b + m) / se, dof), stats.t.cdf((b - m) / se, dof))
        rows.append({'margin_pp': m, 'ci90_low': lo, 'ci90_high': hi,
                     'p_value_tost': p, 'equivalent': bool(lo > -m and hi < m)})
    return pd.DataFrame(rows)


def weighted_disparity_index(df, group_col, protected, reference, rate_col, rate_bins):
    """Weighted Demographic Fairness Index (Eq. 3).

    Contracts are split into interest rate bands (lowest rates first). For
    band i, with weight w_i = exp(-i),

        D_i = (x_ref,i - x_prot,i) / (x_ref,i + x_prot,i)
        WDI = sum_i w_i D_i / sum_i w_i

    The paper's version uses raw counts x = C, which depends on the size of
    each group. The version reported here uses the share of each group in
    the band (x = C / total of the group), so a group that is simply larger
    does not move the index. The index goes from -1 to 1; positive values
    mean the reference group is concentrated in the cheaper bands.

    Returns a dict with the share-based WDI with exponential weights
    ('wdi_shares_exp'), without weights ('wdi_shares_unweighted'), the
    count-based Eq. 3 ('wdi_counts_exp', for comparison) and the table of
    counts by band.
    """
    d = _valid_rows(df, group_col, protected, reference, [rate_col])
    bands = pd.cut(d[rate_col], rate_bins, include_lowest=True)
    c = d.groupby([bands, d[group_col]], observed=False).size().unstack(fill_value=0)
    c = c.reindex(columns=[protected, reference], fill_value=0)
    c_prot, c_ref = c[protected].to_numpy(float), c[reference].to_numpy(float)
    w = np.exp(-np.arange(1, len(c) + 1, dtype=float))

    def index(x_ref, x_prot):
        tot = x_ref + x_prot
        ok = tot > 0
        return np.where(ok, (x_ref - x_prot) / np.where(ok, tot, 1), 0.0), ok

    raw, _ = index(c_ref, c_prot)
    share, ok = index(c_ref / c_ref.sum(), c_prot / c_prot.sum())
    table = c.copy()
    table[f'share_{protected}'] = c_prot / c_prot.sum()
    table[f'share_{reference}'] = c_ref / c_ref.sum()
    table['D_shares'] = share
    table.index = table.index.astype(str)
    table.index.name = 'interest_rate_band'
    return {'wdi_shares_exp': float((w * share).sum() / w.sum()),
            'wdi_shares_unweighted': float(share[ok].mean()),
            'wdi_counts_exp': float((w * raw).sum() / w.sum()),
            'bands': int(len(c)), 'table': table.reset_index()}
