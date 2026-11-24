"""Etapa 4: outros atributos (com e sem controles), localizacao, produto e valor absoluto.

Escada de controles (mesma para todos os atributos; G1 - G2 em p.p. ao ano):
  C0  sem controles (diferenca bruta)
  C1  mesma faixa de score
  C2  mesma celula score x emprestimo/renda (celula da analise principal)
  C3  C2 + numero de parcelas
  C4  C2 + grupo de renda (abaixo/acima de 2 SM)           [nao se aplica a renda]
  C5  C2 + parcelas + grupo de renda + produto
Estimativa = regressao com efeitos fixos de celula, erros agrupados por cliente;
tambem a Eq. 2 (media ponderada das diferencas nas celulas).

Produto e valor absoluto (material para a discussao, atributo renda):
  * mistura de produtos de G1 e G2 e taxa media de cada produto;
  * diferenca de renda dentro de cada produto (celulas da analise principal);
  * diferenca com celulas score x valor absoluto (e + parcelas).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import export as E
from .analysis import ATR, CLI, ROOT, Opcoes, analisar, preparar, resumo_linha

ESCADA = [
    ("C0", "No controls (raw)", dict(medida="nenhuma", controles=()), "bruta"),
    ("C1", "Same risk class", dict(medida="nenhuma", controles=()), "cel"),
    ("C2", "Same score $\\times$ loan-to-income cell", dict(controles=()), "cel"),
    ("C3", "C2 + installments", dict(controles=("parcelas",)), "cel"),
    ("C4", "C2 + income group", dict(controles=("faixa_renda",)), "cel"),
    ("C5", "C2 + installments + income group + product", dict(controles=("faixa_renda", "parcelas", "produto")), "cel"),
]
# Conjunto do artigo (decisao 23/11): sem a distincao por parcelas e sem produto (C3 e C5 ficam
# so na interface); nomes de produto nunca saem nas saidas do artigo (identificariam o banco).
NIVEIS_ARTIGO = ("C0", "C1", "C2", "C4")
ATRIBUTOS = ["sexo", "escolaridade", "idade", "estado_civil", "regiao", "renda_cep", "renda"]


def escada(atributo: str, amostra: str = "janela", B: int = 200, progresso=None, niveis=None) -> pd.DataFrame:
    rows, cache = [], {}
    for cod, lab, kw, tipo in ESCADA:
        if niveis and cod not in niveis:
            continue
        if atributo == "renda" and "faixa_renda" in kw.get("controles", ()):
            kw = dict(kw, controles=tuple(c for c in kw["controles"] if c != "faixa_renda"))
            if cod == "C4":
                continue
            lab = "C2 + installments + product"
        o = Opcoes(amostra=amostra, atributo=atributo, B=B, **kw)
        if o not in cache:
            cache[o] = resumo_linha(analisar(o))
        x = cache[o]
        k = "dif_bruta" if tipo == "bruta" else "dif_celulas_reg"
        rows.append({"atributo": atributo, "amostra": amostra, "nivel": cod, "controles": lab,
                     "contratos": x["contratos"], "clientes": x["clientes"],
                     "contratos_G1": x["contratos_G1"], "contratos_G2": x["contratos_G2"],
                     "taxa_G1": x["taxa_G1"], "taxa_G2": x["taxa_G2"],
                     "dif": x[k], "ic_inf": x[f"{k}_ic_inf"], "ic_sup": x[f"{k}_ic_sup"],
                     "eq2": np.nan if tipo == "bruta" else x["eq2"],
                     "eq2_ic_inf": np.nan if tipo == "bruta" else x.get("eq2_ic_inf"),
                     "eq2_ic_sup": np.nan if tipo == "bruta" else x.get("eq2_ic_sup"),
                     "tag": o.tag()})
        if progresso:
            progresso(f"{atributo} {amostra} {cod}")
    return pd.DataFrame(rows)


def tab_escada_tex(t: pd.DataFrame, amostra: str) -> str:
    t = t[t.amostra == amostra]
    niveis = [c for c, *_ in ESCADA if c in set(t.nivel)]
    head = " & ".join(niveis)
    lin = []
    for a in ATRIBUTOS:
        s = t[t.atributo == a].set_index("nivel")
        if s.empty:
            continue
        r = ATR[a]
        cells = []
        for c in niveis:
            if c in s.index:
                x = s.loc[c]
                cells.append(f"\\makecell{{{x.dif:.1f}\\\\[-2pt]\\tiny[{x.ic_inf:.1f}; {x.ic_sup:.1f}]}}")
            else:
                cells.append("--")
        n = s.iloc[0]
        lin.append(f"\\makecell[l]{{{r['G1_en']} vs.\\\\ {r['G2_en']}}} & {E.n(n.contratos_G1)} / {E.n(n.contratos_G2)} & "
                   + " & ".join(cells) + " \\\\")
    leg = "; ".join(f"{c}: {lab}" for c, lab, *_ in ESCADA if c in niveis)
    return ("\\begin{table}[htbp]\n\\centering\\scriptsize\n"
            "\\caption{Interest-rate difference G1 $-$ G2 (percentage points per year) for each sensitive attribute, "
            f"with progressively stricter comparisons. {leg}. Estimates from regressions "
            "with cell fixed effects; 95\\% CIs clustered by client. Location attributes use borrowers living in the bank's home state; "
            "postcode income = mean income of household heads in the postcode (2022 Census), split at the state median "
            "(the 2022 Census median of the state's postcodes). "
            f"{'Sampling-window sample' if amostra == 'janela' else 'Full sample'}, contracts with registered income.}}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{lc" + "c" * len(niveis) + "}\n\\toprule\n"
            f"G1 vs. G2 & Contracts G1 / G2 & {head} \\\\\n\\midrule\n"
            + "\n".join(lin) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def fig_escada(t: pd.DataFrame, path: Path, amostra: str = "janela"):
    t = t[t.amostra == amostra]
    atrs = [a for a in ATRIBUTOS if a in set(t.atributo)]
    niveis = [c for c, *_ in ESCADA if c in set(t.nivel)]
    cores = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(niveis)))
    fig, ax = plt.subplots(figsize=(7.0, 0.75 * len(atrs) + 1.0))
    for j, a in enumerate(atrs):
        s = t[t.atributo == a].set_index("nivel")
        for i, c in enumerate(niveis):
            if c not in s.index:
                continue
            x = s.loc[c]
            y = len(atrs) - j + (len(niveis) / 2 - i) * 0.11
            ax.errorbar(x.dif, y, xerr=[[x.dif - x.ic_inf], [x.ic_sup - x.dif]], fmt="o", ms=3, color=cores[i],
                        capsize=1.5, lw=1, label=c if j == 0 else None)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_yticks([len(atrs) - j for j in range(len(atrs))],
                  [f"{ATR[a]['G1_en']} vs.\n{ATR[a]['G2_en']}" for a in atrs], fontsize=7)
    ax.set_xlabel("G1 $-$ G2 annual interest rate (percentage points, 95% CI)")
    ax.legend(frameon=False, fontsize=7, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ produto e valor absoluto

def mistura_produtos(amostra="janela") -> pd.DataFrame:
    d, _ = preparar(Opcoes(amostra=amostra))
    g = d.groupby(["PRODUTO", "G1"]).agg(n=("TAXA_ANUAL", "size"), taxa=("TAXA_ANUAL", "mean"),
                                         valor=("VLR_CONTRATADO", "mean"), parcelas=("QTDE_TOTAL_PARCELAS", "mean"))
    t = g.unstack("G1")
    t.columns = [f"{a}_{'G1' if b else 'G2'}" for a, b in t.columns]
    t["pct_dos_G1"] = 100 * t["n_G1"] / t["n_G1"].sum()
    t["pct_dos_G2"] = 100 * t["n_G2"] / t["n_G2"].sum()
    t["taxa_media_produto"] = d.groupby("PRODUTO")["TAXA_ANUAL"].mean()
    return t.sort_values("n_G1", ascending=False).reset_index()


def por_produto(amostra="janela", B=200, min_contratos=2000) -> pd.DataFrame:
    base_ = preparar(Opcoes(amostra=amostra))[0]
    vc = base_["GRUPO_PRODUTO_CONTRATADO"].value_counts()
    rows = []
    for p in vc[vc >= min_contratos].index:
        o = Opcoes(amostra=amostra, produto=p, B=B)
        r = analisar(o)
        x = resumo_linha(r)
        if "dif_celulas_reg" not in x:
            continue
        rows.append({"produto": p, **{k: x[k] for k in ("contratos", "contratos_G1", "contratos_G2", "taxa_G1", "taxa_G2",
                                                        "dif_bruta", "dif_bruta_ic_inf", "dif_bruta_ic_sup",
                                                        "dif_celulas_reg", "dif_celulas_reg_ic_inf", "dif_celulas_reg_ic_sup",
                                                        "eq2")}})
    return pd.DataFrame(rows)


def valor_absoluto(amostra="janela", B=200) -> pd.DataFrame:
    specs = [("Score $\\times$ loan-to-income (main)", Opcoes(amostra=amostra, B=B)),
             ("Score $\\times$ absolute amount", Opcoes(amostra=amostra, medida="valor", B=B)),
             ("Score $\\times$ absolute amount + installments", Opcoes(amostra=amostra, medida="valor", controles=("parcelas",), B=B)),
             ("Score $\\times$ loan-to-income + product", Opcoes(amostra=amostra, controles=("produto",), B=B)),
             ("Score $\\times$ absolute amount + product", Opcoes(amostra=amostra, medida="valor", controles=("produto",), B=B))]
    rows = []
    for lab, o in specs:
        x = resumo_linha(analisar(o))
        rows.append({"comparacao": lab, "amostra": amostra, **{k: x[k] for k in (
            "contratos", "dif_celulas_reg", "dif_celulas_reg_ic_inf", "dif_celulas_reg_ic_sup", "eq1", "eq2")}, "tag": o.tag()})
    return pd.DataFrame(rows)


def tab_produto_tex(mix: pd.DataFrame, pp: pd.DataFrame, va: pd.DataFrame) -> str:
    l1 = [f"{r.PRODUTO.replace('&', '\\&')} & {r.pct_dos_G1:.1f} & {r.pct_dos_G2:.1f} & {r.taxa_G1:.1f} & {r.taxa_G2:.1f} & "
          f"{E.n(r.valor_G1)} & {E.n(r.valor_G2)} \\\\" for r in mix.itertuples()]
    l2 = [f"{r.produto.replace('&', '\\&')} & {E.n(r.contratos)} & {E.ci(r.dif_bruta, r.dif_bruta_ic_inf, r.dif_bruta_ic_sup)} & "
          f"{E.ci(r.dif_celulas_reg, r.dif_celulas_reg_ic_inf, r.dif_celulas_reg_ic_sup)} \\\\" for r in pp.itertuples()]
    l3 = [f"{r.comparacao} & {E.n(r.contratos)} & {E.ci(r.dif_celulas_reg, r.dif_celulas_reg_ic_inf, r.dif_celulas_reg_ic_sup)} & "
          f"{r.eq2:.1f} \\\\" for r in va.itertuples()]
    return ("\\begin{table}[htbp]\n\\centering\\scriptsize\n"
            "\\caption{Income groups (G1 = income below R\\$2,000): product mix, differences within products and with "
            "absolute loan amounts. Material for the discussion. Sampling-window sample, contracts with registered income. "
            "95\\% CIs clustered by client.}\n"
            "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{p{5.2cm}rrrrrr}\n\\toprule\n"
            "\\multicolumn{7}{l}{\\textit{(a) Product mix}} \\\\\n"
            "Product & \\% of G1 & \\% of G2 & Rate G1 & Rate G2 & Loan G1 (R\\$) & Loan G2 (R\\$) \\\\\n\\midrule\n"
            + "\n".join(l1) + "\n\\midrule\n"
            "\\multicolumn{7}{l}{\\textit{(b) Difference G1 $-$ G2 within each product (p.p.)}} \\\\\n"
            "Product & Contracts & Raw & \\multicolumn{4}{l}{Same score $\\times$ loan-to-income cell} \\\\\n\\midrule\n"
            + "\n".join(l2) + "\n\\midrule\n"
            "\\multicolumn{7}{l}{\\textit{(c) Loan size measured in reais, and product controls (p.p.)}} \\\\\n"
            "Comparison & Contracts & Within-cell (regression) & \\multicolumn{4}{l}{Eq.~2} \\\\\n\\midrule\n"
            + "\n".join(l3) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


# ------------------------------------------------------------------ gravacao

def gerar_etapa4(base: Path, progresso=None, amostras=("janela", "completa")) -> Path:
    out = base / "etapa4"
    out.mkdir(parents=True, exist_ok=True)
    log = (lambda m: progresso(0.5, m)) if progresso else None
    partes = []
    for am in amostras:
        for a in ATRIBUTOS:
            try:
                partes.append(escada(a, am, progresso=log, niveis=NIVEIS_ARTIGO))
            except Exception as e:  # atributo sem dados nesta base (ex.: localizacao nos dados sinteticos)
                print(f"[etapa4] {a}/{am} ignorado: {type(e).__name__}")
    t = pd.concat(partes, ignore_index=True)
    t.to_csv(out / "escada_controles_atributos.csv", index=False)
    for am in amostras:
        (out / f"tabela_escada_controles_{am}.tex").write_text(tab_escada_tex(t, am))
        fig_escada(t, out / f"fig_escada_controles_{am}.pdf", am)
    mix = mistura_produtos()
    pp = por_produto()
    # nomes de produto -> letras (A = mais contratos); o mapa fica so em data/derived (privado)
    mapa = {p: f"Product {chr(65 + i)}" for i, p in enumerate(mix.sort_values("n_G1", ascending=False).PRODUTO)}
    mapa["OUTROS"] = "Other products"
    pd.Series(mapa, name="rotulo").rename_axis("produto").to_csv(ROOT / "data" / "derived" / "mapa_produtos_PRIVADO.csv")
    mix["PRODUTO"] = mix["PRODUTO"].map(mapa)
    pp["produto"] = pp["produto"].map(lambda p: mapa.get(p, "Other products"))
    va = pd.concat([valor_absoluto(am) for am in amostras], ignore_index=True)
    mix.to_csv(out / "produto_mistura_renda.csv", index=False)
    pp.to_csv(out / "produto_diferenca_dentro.csv", index=False)
    va.to_csv(out / "valor_absoluto_e_produto.csv", index=False)
    (out / "tabela_produto_valor_absoluto.tex").write_text(tab_produto_tex(mix, pp, va[va.amostra == "janela"]))
    return out
