# LoanDiscrimination

Code and simulated data for the paper **"The Cost of Fairness: Examining Hidden Discrimination in Bank Loan Practices"** (A. C. B. Garcia, M. G. P. Garcia, R. Rigobon).

The repository measures whether two demographic groups with the same credit score are charged different interest rates, and estimates what it would cost a bank to reduce that difference.

## What is here and what is not

The paper has two parts. The repository covers the first one with a simulated dataset, and reproduces the analysis of the second one on a synthetic dataset with the same structure.

| Part of the paper | Data | In this repository? |
|---|---|---|
| Illustration of the method (fairness metrics, cost of fairness, Figure 4) | `data/Simulated_Dataset.csv` (gender example) | Yes: data, code and results (`run_analysis.py`) |
| Case study of a Brazilian bank (Tables 4 and 5, Figures 6 to 8) | Confidential loan records | The bank data cannot be shared. The same analysis runs on `data/Synthetic_Income_Dataset.csv`, an artificial dataset with the columns the method needs (`run_income_example.py`) |

No bank data, and no result computed from bank data, is stored in this repository.

## Results

The results on the simulated dataset are in [`outputs/results.md`](outputs/results.md). The page opens directly on GitHub, with tables and figures, so nothing needs to be installed to read it.

Summary of what the report shows:

- **Dataset.** 10,000 simulated contracts, 5,037 for women and 4,963 for men, with the same average credit score and default rate in both groups.
- **Price gap.** Women are charged 127.9% a year on average and men 119.8%.
- **Same cell, different price.** Comparing women and men in the same comparable cell (credit score bin x loan amount bin x number of installments; 43 valid cells, 9,449 contracts), women pay 8.27 percentage points more (Eq. 2, weighted by cell size; 9.05 in the simple mean of Eq. 1). Women pay more in all 43 cells.
- **Significance.** In the regression with cell fixed effects and robust standard errors, the gap is 8.27 p.p. (95% CI 8.12 to 8.41, p < 0.0001). The equivalence test (TOST) rejects equality at margins of 1, 2 and 5 p.p.
- **Weighted Demographic Fairness Index (WDI).** 0.953 with group shares and weights e^-i (0.146 without weights), on a scale from -1 to 1 where 0 means no disparity: men are concentrated in the lowest interest rate bands.
- **Cost of fairness.** Closing 100% of the within-cell gap for clients with credit score up to 0.4 costs R$ 53,744, or 1.82% of the term interest income of the portfolio. Closing it for every client costs 2.58%. The cost grows linearly with the share of the gap closed.

These numbers describe simulated data built to contain discrimination. They show how the method works and are not findings about any real lender.

## Income case study on synthetic data

`data/Synthetic_Income_Dataset.csv` has 10,000 artificial personal-loan contracts (6,000 clients), built by `src/SyntheticIncome.py` with a fixed seed. By construction, borrowers earning less than R$ 2,000 a month (about two minimum wages) have the same credit-score and loan-to-income distributions as the other borrowers but pay 18 percentage points more per year. Default depends only on the credit score up to 0.4, and is somewhat higher for lower-income borrowers above it. Sex, age and marital status have no effect on the rate; schooling and age are correlated with income.

```bash
python run_income_example.py            # writes outputs/income_example/
python run_income_example.py --regenerate  # rebuilds the dataset first
```

The report [`outputs/income_example/results_income.md`](outputs/income_example/results_income.md) shows, step by step: the distribution of the data, Table 4 (rate by credit-score range and income group), Eq. 1 and Eq. 2 in score x loan-to-income cells, the fixed-effects test with standard errors clustered by client (which recovers the 18 p.p. built into the data), the WDI, the cost of fairness with interest computed over each loan's term (Figure 6 and Table 5), and Figures 4, 7 and 8.

| Column | Meaning |
|---|---|
| `ContractID`, `ClientID` | Contract and client identifiers (a client may have several contracts) |
| `CreditScore` | Probability of default estimated by the lender, from 0 (best) to 1 (worst) |
| `MonthlyIncome`, `IncomeGroup` | Monthly income (R$) and group: `< 2 MW` or `>= 2 MW` (threshold R$ 2,000) |
| `LoanAmount`, `LoanToIncome` | Amount lent (R$) and loan amount divided by monthly income |
| `Installments`, `PaidInstallments` | Number of installments, and installments paid (fewer than the total for defaulted contracts) |
| `MonthlyRate`, `InterestRate` | Monthly rate (%) and annual rate (%), with ((1 + monthly/100)^12 - 1) x 100 |
| `Sex`, `Age`, `SchoolingCode`, `Partner` | Borrower attributes (schooling code: odd values 3 to 29, higher = more years of schooling) |
| `Default` | 1 if the contract defaulted, 0 otherwise |

These numbers describe artificial data built to contain an income penalty. They show how the method works and are not findings about any lender.

## Data

`data/Simulated_Dataset.csv` contains 10,000 simulated loan contracts with gender-based price discrimination built in. It illustrates the method. The bank dataset analyzed in the paper is confidential and is not included here.

| Column | Meaning |
|---|---|
| `LoanAmount` | Amount lent |
| `CreditScore` | Probability of default estimated by the lender, from 0 (best) to 1 (worst) |
| `Installments` | Number of installments |
| `Gender` | `Male` or `Female` |
| `InterestRate` | Annual interest rate, in percent |
| `Default` | 1 if the client defaulted, 0 otherwise |

## How to run

Requires Python 3.9 or later.

```bash
git clone https://github.com/CristinaBicharra/LoanDiscrimination.git
cd LoanDiscrimination
pip install -r requirements.txt
python run_analysis.py
```

The script writes everything to `outputs/`. Open `outputs/results.md` to read the report. A copy of the outputs is already in the repository, so the results can be inspected without running anything.

To use the notebook instead, open `Notebook/Fairness_Main.ipynb` in Jupyter and run all cells. It calls the same script and shows the report inline.

## What the analysis produces

| File in `outputs/` | Content |
|---|---|
| `results.md` | Full report: data, Eq. 1 and Eq. 2, fixed-effects test and TOST, WDI, cost of fairness |
| `fig_interest_rate_histogram.png` | Number of contracts by interest rate and gender (Figure 4 of the paper) |
| `fig_interest_rate_by_score.png` | Interest rate against credit score, by gender |
| `fig_disparity_by_score_bin.png` | Within-cell rate gap (Eq. 2) in each credit score bin, with 95% interval |
| `fig_cost_of_fairness.png` | Cost, in % of term interest income, of closing a share of the gap, by score cutoff |
| `cells.csv` | Every comparable cell: contracts and mean rate of each group, gap, whether it is valid |
| `disparity_by_score_bin.csv` | Eq. 1 and Eq. 2 within each credit score bin |
| `wdi_bands.csv` | Contracts and share of each group in each interest rate band (input of the WDI) |
| `cost_of_fairness.csv` | Cost for each share of the gap closed and each score cutoff, with the Eq. 4 comparison |

## Method

The protected group (women) is compared with the reference group (men). Rates are annual, in percent; a gap is protected minus reference, in percentage points (p.p.).

1. **Comparable cells.** Contracts are grouped into cells of similar credit risk and loan terms. In the paper the cells are credit score bin x (loan amount / income). The simulated dataset has no income, so here the cells are credit score bin x loan amount bin (R$ 2,000 wide) x number of installments. A cell is valid only with at least 30 contracts of each group; the report gives how many cells and contracts were dropped.
2. **Eq. 1 and Eq. 2.** The gap of each valid cell is the mean rate of women minus the mean rate of men. Eq. 1 is the simple mean of the cell gaps; Eq. 2 weights each cell by its number of contracts.
3. **Significance test.** Rate regressed on a female dummy with cell fixed effects, with robust standard errors: clustered by client when a client id exists (as in the paper's bank data), HC1 otherwise (the simulated data has one contract per client). The report gives the coefficient, 95% interval and p-value. A non-significant difference is not evidence of equal rates: equality is claimed only through an equivalence test (TOST) with margins of 1, 2 and 5 p.p.
4. **Weighted Demographic Fairness Index (Eq. 3).** For each interest rate band i (cheapest first), D_i = (x_men - x_women) / (x_men + x_women), with x the *share* of each group's contracts in the band, averaged with weights e^-i and also without weights. The raw-count version of Eq. 3 changes with the relative size of the groups, so it is shown only for comparison.
5. **Cost of fairness.** Interest is computed over the loan term with the Price amortization table: monthly rate = (1 + annual rate)^(1/12) - 1, installment = L m / (1 - (1 + m)^-n), interest = n x installment - L, using each contract's number of installments. The policy lowers women's rate by alpha x (cell gap) in the cells where women pay more, for contracts with credit score at or below a cutoff. The cost is the interest income the lender gives up, in R$ and as % of the portfolio's term interest income, for alpha from 0 to 100% and cutoffs 0.1, 0.2, 0.4 and all scores. The simulated data does not record how many installments a defaulting client paid, so defaulted contracts earn no interest here (`term_interest_income` accepts that column when it exists). The paper's Eq. 4 (annual rate x principal for repaid contracts, minus defaulted principal) ignores the loan term and is kept only as a comparison.

## Repository structure

```
LoanDiscrimination/
├── data/
│   └── Simulated_Dataset.csv
├── Notebook/
│   └── Fairness_Main.ipynb
├── outputs/                      results of run_analysis.py
├── src/
│   ├── CostOfFairness.py         term interest (Price table) and cost curve
│   ├── DiscriminationMetrics.py  cells, Eq. 1-3, fixed-effects test, TOST
│   └── VisualizationPKG.py       plots
├── run_analysis.py               runs the whole analysis
├── requirements.txt
└── README.md
```

## Using the functions on other data

```python
import pandas as pd
from src.DiscriminationMetrics import cell_table, rate_gap_metrics, fixed_effects_test, tost
from src.CostOfFairness import cost_curve

df = pd.read_csv("data/Simulated_Dataset.csv")
df["ScoreBin"] = pd.cut(df["CreditScore"], [0, 0.1, 0.2, 0.4, 0.6, 0.9], include_lowest=True)
df["LoanBin"] = pd.cut(df["LoanAmount"], [0, 4000, 8000, 12000, float("inf")])
keys = ["ScoreBin", "LoanBin", "Installments"]
group = ("Gender", "Female", "Male")

print(rate_gap_metrics(cell_table(df, *group, "InterestRate", keys)))      # Eq. 1, Eq. 2, cells
fe = fixed_effects_test(df, *group, "InterestRate", keys, cluster_col=None)  # pass the client id column if any
print(fe, tost(fe["b"], fe["se"], fe["dof"]))

curve = cost_curve(df, *group, "InterestRate", keys, "ScoreBin",
                   "LoanAmount", "Installments", "Default")
print(curve[curve.alpha == 1.0])
```

## Contact

Ana Cristina Bicharra Garcia, cristina.bicharra@uniriotec.br
