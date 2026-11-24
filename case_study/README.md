# Case study code

This folder contains the code used for the case study of the paper (Section 4): the analysis of the
personal-credit contracts of a Brazilian bank. The bank data are confidential and are **not** in this
repository; the code runs on the public synthetic dataset (`../data/Synthetic_Income_Dataset.csv`), which has
the same structure, so reviewers can execute every step.

## Run on the synthetic data

```bash
cd case_study
pip install pandas numpy matplotlib statsmodels pyfixest xgboost scikit-learn streamlit pyarrow
python3 scripts/prepare_synthetic.py      # builds data/derived/contratos_cp_id.parquet
python3 scripts/gerar_conjunto.py         # all tables (LaTeX) and figures (PDF) -> outputs/conjunto_artigo/
python3 scripts/copiar_para_paper.py      # the tables and figures used in the paper -> paper/tables, paper/figures
python3 scripts/correcoes_tabelas.py      # label corrections applied to the paper tables (see below)
streamlit run app/app.py                  # interactive interface, http://localhost:8766
```

The synthetic data have no contract dates, products or location; those parts are filled with neutral
values or skipped. The numbers obtained on the synthetic data are not the numbers of the paper.

## What the code does (as in the paper)

| Step | Module | Method |
|---|---|---|
| Contract base | `src/cf/build.py` | One record per contract identifier (first monthly observation); personal credit only; borrowers aged 18 or more; sampling window July 2019 to June 2020 |
| Groups and cells | `src/cf/analysis.py` | Lower-income group: registered income below R$2,000 (about two minimum wages). Cells: the bank's 12 credit-risk classes x 22 loan-to-income bins |
| Interest-rate difference | `src/cf/analysis.py` | OLS of the annual rate on a group indicator with cell fixed effects, standard errors clustered by borrower; t test within each risk class with Holm correction |
| Fairness metrics | `src/cf/analysis.py` | Eq. 1 and Eq. 2 over cells with both groups, 95% CI from a borrower-level bootstrap; WDI (Eq. 3) with the no-disparity reference (k-1)/(k+1) |
| Default | `src/cf/default_cost.py` | 24-month client default indicator by group and risk class; t tests with Holm correction |
| Cost of fairness | `src/cf/default_cost.py` | Reduce the lower-income rate by a share of the cell difference; interest by Price amortization; expected revenue with PD from the score or observed by group |
| Other attributes | `src/cf/etapa4.py` | Same comparison for sex, age, schooling, marital status and location, with progressively stricter controls |
| Robustness without bins | `src/cf/xgb.py` | XGBoost trained on one income group and applied to the other (counterfactual), grouped cross-validation, bootstrap CI |
| Tables and figures | `src/cf/export*.py`, `src/cf/paper_tables.py` | LaTeX tables and PDF figures named after the options used |

All options (sample, missing income, loan-size measure, controls, attribute, threshold, score classes, cutoff)
are parameters of `Opcoes` in `src/cf/analysis.py` and can be changed in the interface.

## Corrections applied to the paper tables

`scripts/correcoes_tabelas.py` applies to the copies in `paper/tables` the corrections made in the paper after the
results were frozen; none of them changes a result: (1) public servants are the borrowers with job codes 1, 3 and 4
(municipal, state or federal government employees), so the labels of the public-servant rows in the descriptive
tables are swapped; (2) the paper does not compare income groups within each product, so those rows are removed from
the product table; (3) the loan-to-income column of the summary table shows only quartiles, because a few registered
incomes are close to zero; (4) "R$" is escaped in the captions.

## Bank-specific settings

The file names of the bank data, the rules that map product names to loan types and the location parameters
are kept in `config/private_settings.json`, which is not published (it would identify the bank). The file
`config/private_settings.example.json` shows its structure. Code comments are in Portuguese.
