# LoanDiscrimination

Code and simulated data for the paper **"The Cost of Fairness: Examining Hidden Discrimination in Bank Loan Practices"** (A. C. B. Garcia, M. G. P. Garcia, R. Rigobon).

The repository measures whether two demographic groups with the same credit score are charged different interest rates, and estimates what it would cost a bank to reduce that difference.

## What is here and what is not

The paper has two parts, and this repository covers the first one.

| Part of the paper | Data | In this repository? |
|---|---|---|
| Illustration of the method (fairness metrics, cost of fairness, Figure 4) | Simulated dataset | Yes: data, code and results |
| Case study of a Brazilian bank (Tables 1 to 4, Figures 7 to 9) | Confidential loan records | No: the data cannot be shared, so those results appear only in the paper |

No bank data, and no result computed from bank data, is stored in this repository.

## Results

The results on the simulated dataset are in [`outputs/results.md`](outputs/results.md). The page opens directly on GitHub, with tables and figures, so nothing needs to be installed to read it.

Summary of what the report shows:

- **Dataset.** 10,000 simulated contracts, 5,037 for women and 4,963 for men, with the same default rate in both groups.
- **Price gap.** Women are charged 127.9% a year on average and men 119.8%.
- **Same credit score, different price.** Comparing only clients in the same credit score bin, women pay 8.5 percentage points more (weighted average; 11.2 in the simple average). The difference is statistically significant in all 12 bins.
- **Weighted Demographic Fairness Index (WDI).** 0.95 on a scale from -1 to 1, where 0 means no disparity: men are concentrated in the lowest interest rate ranges.
- **Cost of fairness.** Removing 20% of the gap for the four best credit score bins costs the lender about 18,800. Removing the whole gap for every client costs about 1.24 million, or 3.6% of the interest income from paid contracts.

These numbers describe simulated data built to contain discrimination. They show how the method works and are not findings about any real lender.

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
| `results.md` | Full report: descriptive statistics, fairness metrics, cost of fairness |
| `fig_interest_rate_histogram.png` | Interest rate distribution by gender (Figure 4 of the paper) |
| `fig_interest_rate_by_score.png` | Interest rate against credit score, by gender |
| `fig_disparity_by_score_bin.png` | Interest rate disparity in each credit score bin |
| `fig_cost_of_fairness.png` | Cost of removing a share of the interest rate gap |
| `disparity_by_score_bin.csv` | Disparity and significance test for each credit score bin |
| `cost_of_fairness.csv` | Cost for each share of the gap removed and each set of bins covered |

## Method

1. **Comparison groups.** Clients are grouped into credit score bins, so that each comparison is between borrowers with similar creditworthiness.
2. **Fairness metrics.**
   - Simple average interest rate comparison: mean of the rate differences across bins.
   - Weighted-average interest rate comparison: the same mean, weighted by the number of clients in each bin.
   - Weighted Demographic Fairness Index (WDI): compares how many clients of each group fall in each interest rate range, giving more weight to the lowest rates. It ranges from -1 to 1, and 0 means no disparity.
3. **Cost of fairness.** For each credit score bin, the cost of lowering the rate of the discriminated group is

   `share of paid contracts x average loan x rate reduction x number of contracts`

   with the rate reduction expressed as a fraction (10 percentage points enter as 0.10).

## Repository structure

```
LoanDiscrimination/
├── data/
│   └── Simulated_Dataset.csv
├── Notebook/
│   └── Fairness_Main.ipynb
├── outputs/                      results of run_analysis.py
├── src/
│   ├── CostOfFairness.py         cost of reducing the rate gap
│   ├── DiscriminationMetrics.py  fairness metrics and significance tests
│   └── VisualizationPKG.py       plots
├── run_analysis.py               runs the whole analysis
├── requirements.txt
└── README.md
```

## Using the functions on other data

```python
import pandas as pd
from src.DiscriminationMetrics import calculate_equality_of_opportunity
from src.CostOfFairness import calculate_fairness_cost

df = pd.read_csv("data/Simulated_Dataset.csv")
df["CreditScoreBin"] = pd.cut(df["CreditScore"], bins=[0, 0.1, 0.2, 0.4, 0.6, 0.9], include_lowest=True)

result = calculate_equality_of_opportunity(df, "Gender", "Female", "Male", "InterestRate", "CreditScoreBin")
print(result["weighted"])

cost = calculate_fairness_cost(df, "LoanAmount", "CreditScoreBin", "Gender", "Male", "Female",
                               "InterestRate", fairness_degree=0.2, max_category=2, Default_var="Default")
print(cost)
```

## Contact

Ana Cristina Bicharra Garcia, cristina.bicharra@uniriotec.br
