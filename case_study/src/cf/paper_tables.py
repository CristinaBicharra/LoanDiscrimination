"""Tabelas descritivas (Tabelas 2 e 3 do artigo) e figura das taxas por faixa de score
para cada atributo (substitui a Fig. 7), sempre na amostra da analise principal.

Codigos (os mesmos do codigo de 2023, que reproduzem a Tabela 3 original):
  servidor publico = FUNC_PUB_ATIVIDADE == 5; salario depositado no banco = IND_FOLHA_BCO == 1;
  com conjuge = COD_ESTADO_CIVIL em {1, 5}; mulher = SEXO == 1.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import export as E
from .analysis import ATR, Opcoes, analisar, preparar


def _pm(s, d=1):
    return f"{s.mean():,.{d}f} $\\pm$ {s.std():,.{d}f}"


def tabela2(o: Opcoes = Opcoes()) -> tuple[pd.DataFrame, str]:
    d, _ = preparar(o)
    cols = {"Score": d["Score"], "IR": d["TAXA_ANUAL"], "Loan": d["VLR_CONTRATADO"], "Income": d["RENDA"],
            "Loan/Income": d["VLR_CONTRATADO"] / d["RENDA"].where(d["RENDA"] > 0),
            "Installments": d["QTDE_TOTAL_PARCELAS"], "Age": d["IDADE"],
            "School code": d["COD_ESCOLARIDADE"], "Public job": (d["FUNC_PUB_ATIVIDADE"] == 5).astype(float),
            "DD": (d["IND_FOLHA_BCO"] == 1).astype(float)}
    t = pd.DataFrame({k: v.describe(percentiles=[0.25, 0.5, 0.75]) for k, v in cols.items()}).T
    t = t[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    fmt = {"Score": 2, "IR": 1, "Loan": 0, "Income": 0, "Loan/Income": 2, "Installments": 0, "Age": 0,
           "School code": 0, "Public job": 2, "DD": 2}
    lin = []
    for stat, lab in (("mean", "Mean"), ("std", "Std"), ("min", "Min"), ("25%", "25\\%"), ("50%", "50\\%"),
                      ("75%", "75\\%"), ("max", "Max")):
        lin.append(f"\\textbf{{{lab}}} & " + " & ".join(f"{t.loc[c, stat]:,.{fmt[c]}f}" for c in cols) + " \\\\")
    tex = ("\\begin{table}[htbp]\n\\centering\\scriptsize\n"
           "\\caption{\\add{Summary statistics of the personal-credit contracts in the main analysis sample "
           f"({len(d):,} contracts of {d['CHAVE_ALEATORIA'].nunique():,} borrowers aged 18 or more, started between July 2019 "
           "and June 2020, with registered income). Score = credit score (higher = riskier); IR = annual interest rate (\\%); "
           "Loan = loan amount (R\\$); Income = registered monthly income (R\\$); School code = schooling code in the "
           "bank's records (odd codes, higher = more schooling); Public job = government employee; DD = wage deposited "
           "in the bank.}}\\label{tab:2}\n"
           "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n"
           "\\textbf{Stat} & " + " & ".join(f"\\textbf{{{c}}}" for c in cols) + " \\\\\n\\midrule\n"
           + "\n".join(lin) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    return t, tex


def tabela3(o: Opcoes = Opcoes()) -> tuple[pd.DataFrame, str]:
    d, _ = preparar(o)
    d = d.copy()
    grupos = {
        "SEX": ("SEXO", {1: "Female", 2: "Male"}),
        "AGE": ("FX_IDADE", None),
        "SCHOOLING (bank code)": ("COD_ESCOLARIDADE", None),
        "PARTNER": ("PARC", None),
        "JOB: PUBLIC SERVANT": ("PUB", None),
        "WAGE DEPOSITED IN THE BANK": ("DD", None),
        "MONTHLY INCOME": ("GR", None),
    }
    d["FX_IDADE"] = np.where(d["IDADE"] >= 60, "60 or more", "18--59")
    d["PARC"] = np.where(d["COD_ESTADO_CIVIL"].isin([1, 5]), "Yes", "No")
    d["PUB"] = np.where(d["FUNC_PUB_ATIVIDADE"] == 5, "Yes", "No")
    d["DD"] = np.where(d["IND_FOLHA_BCO"] == 1, "Yes", "No")
    d["GR"] = np.where((d["RENDA"] <= d["LIMITE"]) if o.limite_inclusivo else (d["RENDA"] < d["LIMITE"]),
                       "Below R$2,000", "R$2,000 or more")
    rows, lin = [], []
    for titulo, (col, mapa) in grupos.items():
        lin.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{titulo}}}}} \\\\")
        g = d.groupby(col)
        for k, s in g:
            lab = mapa.get(k, k) if mapa else (f"{int(k)}" if isinstance(k, (int, float, np.floating)) else k)
            rows.append({"grupo": titulo, "categoria": lab, "n": len(s)})
            lin.append(f"{lab} & {len(s):,} & {_pm(s['Score'], 2)} & {_pm(s['VLR_CONTRATADO'], 0)} & "
                       f"{_pm(s['QTDE_TOTAL_PARCELAS'], 0)} & {_pm(s['TAXA_ANUAL'], 1)} \\\\")
    tex = ("{\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{lrcccc}\n"
           "\\caption{\\add{Personal-credit contracts in the main analysis sample by borrower characteristic "
           "(mean $\\pm$ standard deviation). Loan in R\\$; IR = annual interest rate (\\%).}}\\label{tab:3}\\\\\n"
           "\\toprule\n\\textbf{Category} & \\textbf{Qty} & \\textbf{Score} & \\textbf{Loan} & "
           "\\textbf{Installments} & \\textbf{Annual IR (\\%)} \\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endlastfoot\n"
           + "\n".join(lin) + "\n\\end{longtable}\n}\n")
    return pd.DataFrame(rows), tex


def fig_atributos(path: Path, atributos=("renda", "escolaridade", "idade", "estado_civil", "sexo", "renda_cep")):
    titulos = {"renda": "Monthly income", "escolaridade": "Schooling", "idade": "Age",
               "estado_civil": "Marital status", "sexo": "Sex", "renda_cep": "Postcode income"}
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 8.4), sharex=True, sharey=True)
    for ax, a in zip(axs.ravel(), atributos):
        try:
            r = analisar(Opcoes(atributo=a, B=0))
        except Exception:
            ax.set_visible(False)
            continue
        if "por_score" not in r:  # atributo sem os dois grupos nesta base
            ax.set_visible(False)
            continue
        t = r["por_score"]
        x = np.arange(len(t))
        for g, dx in (("G2", -0.1), ("G1", 0.1)):
            se = t[f"dp_{g}"] / np.sqrt(t[f"n_{g}"])
            ax.errorbar(x + dx, t[f"taxa_{g}"], yerr=1.96 * se, fmt="o-", ms=2.5, lw=1, capsize=1.5,
                        color=E.COR[g], label=ATR[a][g + "_en"])
        ax.set_title(titulos[a], fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(x, t["faixa_score"], rotation=60, ha="right", fontsize=6)
    for ax in axs[:, 0]:
        ax.set_ylabel("Mean annual interest rate (%)", fontsize=7.5)
    fig.supxlabel("Credit-risk class (score range; higher = riskier)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def troca_de_grupo() -> dict:
    """Com o limite de 2 SM na data do contrato, quantos clientes mudam de grupo sem mudar de renda."""
    from .analysis import CLI
    d, _ = preparar(Opcoes(limite_renda="2SM"))
    g = d.groupby(CLI)
    troca = (g["G1"].nunique() > 1) & (g["RENDA"].nunique() == 1)
    return {"clientes_trocam_mesma_renda": int(troca.sum()),
            "contratos_desses_clientes": int(d[CLI].isin(troca[troca].index).sum()),
            "clientes": int(d[CLI].nunique())}


def gerar(base: Path) -> None:
    out = base / "artigo_descritivas"
    out.mkdir(parents=True, exist_ok=True)
    t2, x2 = tabela2()
    t2.to_csv(out / "tabela2_descritivas.csv")
    (out / "tabela2_descritivas.tex").write_text(x2)
    t3, x3 = tabela3()
    t3.to_csv(out / "tabela3_por_caracteristica.csv", index=False)
    (out / "tabela3_por_caracteristica.tex").write_text(x3)
    fig_atributos(out / "fig7_taxa_por_score_atributos.pdf")
    import json
    (base / "troca_de_grupo.json").write_text(json.dumps(troca_de_grupo(), indent=2))
