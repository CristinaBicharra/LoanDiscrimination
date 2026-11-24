"""Builds the analysis base from the public synthetic dataset, so the case-study code can run without the
confidential bank data.

    python3 scripts/prepare_synthetic.py      (from the case_study folder)

Reads ../data/Synthetic_Income_Dataset.csv and writes data/derived/contratos_cp_id.parquet with the column
names the code expects (the bank's column names). Fields that the synthetic data does not have (contract
dates, product, location, job type) are filled with neutral values; the location analyses are skipped.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "data" / "Synthetic_Income_Dataset.csv"
OUT = ROOT / "data" / "derived"
OUT.mkdir(parents=True, exist_ok=True)

s = pd.read_csv(SRC)
rng = np.random.default_rng(2026)
inicio = pd.Timestamp("2019-07-01") + pd.to_timedelta(rng.integers(0, 366, len(s)), unit="D")
d = pd.DataFrame({
    "COD_CONTRATO_MASCARADO": s["ContractID"].astype(str),
    "CHAVE_ALEATORIA": s["ClientID"].astype(str),
    "GRUPO_PRODUTO_CONTRATADO": "PERSONAL CREDIT",
    "DESCR_PRODUTO_GESTAO": "PERSONAL CREDIT",
    "DATA_INICIO_CONTRATO": inicio,
    "VLR_CONTRATADO": s["LoanAmount"].astype(float),
    "QTDE_TOTAL_PARCELAS": s["Installments"].astype(float),
    "NUM_PARCELA_PAGA": s["PaidInstallments"].astype(float),
    "TAXA_MENSAL_OPERACAO": s["MonthlyRate"].astype(float),
    "MAU_24": s["Default"].astype(float),
    "IDADE": s["Age"].astype(float),
    "SEXO": np.where(s["Sex"] == "Female", 1.0, 2.0),
    "COD_ESCOLARIDADE": s["SchoolingCode"].astype(float),
    "COD_ESTADO_CIVIL": np.where(s["Partner"] == "Yes", 1.0, 6.0),
    "RENDA_CADASTRAL": s["MonthlyIncome"].astype(float),
    "FUNC_PUB_ATIVIDADE": 0.0,
    "IND_FOLHA_BCO": 0.0,
    "Score": s["CreditScore"].astype(float),
})
d["CLASSE"] = "Credito_Pessoal"
d["NA_JANELA"] = True
d["TAXA_ANUAL"] = ((1 + d["TAXA_MENSAL_OPERACAO"] / 100) ** 12 - 1) * 100
d["SEM_RENDA"] = ~(d["RENDA_CADASTRAL"] > 0)
for unidade in ("id", "id_valor"):
    nome = "contratos_cp_id.parquet" if unidade == "id" else "contratos_cp.parquet"
    d.to_parquet(OUT / nome, index=False)
(OUT / "build_stats.json").write_text(json.dumps({"id": {"fonte": "synthetic", "contratos": len(d)}}, indent=2))
print("written:", OUT / "contratos_cp_id.parquet", len(d), "contracts")
