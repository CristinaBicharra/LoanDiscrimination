"""Synthetic dataset for the income case study of the paper.

The bank data analyzed in the paper are confidential. This module builds a
simplified, fully artificial dataset with the columns the method needs, so
that the case-study analysis (Tables 4 and 5, Figures 4, 6, 7 and 8 of the
paper) can be run and understood without the bank data.

Built-in pattern (by design, NOT estimated from the bank):
* borrowers earning less than R$ 2,000 a month have the SAME distribution of
  credit score and loan-to-income ratio as the other borrowers;
* they are charged a higher annual interest rate (INCOME_PENALTY_PP p.p.);
* default depends only on the credit score up to 0.4; above 0.4 lower-income
  borrowers default somewhat more, as in Figure 8 of the paper.

Sex, age and marital status have no direct effect on the rate. Schooling and
age are correlated with income, so they show a gap only through income.

Every number is drawn from simple distributions with a fixed seed; nothing
comes from the bank's records.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INCOME_THRESHOLD = 2000.0     # two minimum wages (R$), as in the paper
INCOME_PENALTY_PP = 18.0      # extra annual rate (p.p.) charged to income < R$ 2,000
N_CONTRACTS = 10_000
N_CLIENTS = 6_000
SEED = 20261121
INSTALLMENT_OPTIONS = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24])
INSTALLMENT_WEIGHTS = np.array([25, 15, 12, 10, 9, 7, 6, 5, 6, 3, 2], float)


def make_synthetic_income(n_contracts: int = N_CONTRACTS, n_clients: int = N_CLIENTS,
                          seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- clients: attributes fixed for all contracts of the same client
    income = np.exp(rng.normal(np.log(INCOME_THRESHOLD), 0.75, n_clients))
    income = np.round(np.clip(income, 400, 40_000), 2)
    score = np.clip(rng.beta(1.3, 6.5, n_clients), 0.01, 0.90).round(4)
    sex = rng.choice(['Female', 'Male'], n_clients, p=[0.6, 0.4])
    age = np.round(rng.normal(37, 10, n_clients) + 2 * np.log(income / INCOME_THRESHOLD))
    senior = rng.random(n_clients) < 0.12                 # about 12% aged 60 or more
    age = np.clip(np.where(senior, rng.uniform(60, 82, n_clients).round(), age), 18, 85)
    z = (np.log(income) - np.log(INCOME_THRESHOLD)) / 0.75 + rng.normal(0, 1, n_clients)
    schooling = (13 + 2 * np.round(np.clip(2.2 * z, -5, 8))).astype(int)  # odd codes 3..29
    partner = rng.choice(['Yes', 'No'], n_clients, p=[0.32, 0.68])

    # ---- contracts
    client = rng.integers(0, n_clients, n_contracts)
    lti = np.clip(np.exp(rng.normal(np.log(0.25), 0.8, n_contracts)), 0.02, 3.0)
    loan = np.maximum(50, np.round(lti * income[client], -1))
    installments = rng.choice(INSTALLMENT_OPTIONS, n_contracts,
                              p=INSTALLMENT_WEIGHTS / INSTALLMENT_WEIGHTS.sum())
    s = score[client]
    low = income[client] < INCOME_THRESHOLD

    base = 50 + 300 * s - 200 * s ** 2                   # risk-based price
    term = -1.5 * (installments - 1)                    # longer loans slightly cheaper
    annual = base + term + INCOME_PENALTY_PP * low + rng.normal(0, 12, n_contracts)
    annual = np.clip(annual, 12, 300).round(2)
    monthly = (((1 + annual / 100) ** (1 / 12) - 1) * 100).round(4)

    p_default = 1 / (1 + np.exp(-(-2.6 + 5.5 * s))) + 0.08 * (low & (s > 0.4))
    default = (rng.random(n_contracts) < p_default).astype(int)
    paid = np.where(default == 1, np.floor(rng.random(n_contracts) * installments), installments)

    df = pd.DataFrame({
        'ContractID': np.arange(1, n_contracts + 1),
        'ClientID': client + 1,
        'CreditScore': s,
        'MonthlyIncome': income[client],
        'IncomeGroup': np.where(low, '< 2 MW', '>= 2 MW'),
        'LoanAmount': loan,
        'LoanToIncome': (loan / income[client]).round(4),
        'Installments': installments,
        'MonthlyRate': monthly,
        'InterestRate': annual,
        'Sex': sex[client],
        'Age': age[client].astype(int),
        'SchoolingCode': schooling[client],
        'Partner': partner[client],
        'Default': default,
        'PaidInstallments': paid.astype(int),
    })
    return df


if __name__ == '__main__':
    d = make_synthetic_income()
    print(d.describe().round(2).T)
