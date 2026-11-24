"""Etapa 3: tabelas LaTeX, figuras PDF e CSV de inadimplencia (MAU_24) e custo da equidade."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import export as E  # configura matplotlib (Agg, fontes)
from .analysis import ATR, Opcoes
from .default_cost import ALPHAS, custo, inadimplencia

VERSOES = {"score": "Default implied by the credit score",
           "observado": "Observed default of each group"}


def _rot(o: Opcoes):
    return ATR[o.atributo]



def _pf(v):
    import math
    return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else ("$<$0.001" if v < 0.001 else f"{v:.2f}")


# ------------------------------------------------------------------ (a)

def tab_inad_tex(r: dict, o: Opcoes) -> str:
    a = r["agregado"]
    rot = _rot(o)
    nome = {"todos": "All scores"}
    lin = []
    for _, x in a.iterrows():
        sub = nome.get(x.subamostra, x.subamostra.replace("<=", "$\\leq$").replace(">", "$>$").replace("score", "Score"))
        lin.append(f"{sub} & {'Contract' if x.nivel == 'contrato' else 'Client'} & {E.n(x.n_G1)} & {E.n(x.n_G2)} & "
                   f"{x.G1:.1f} & {x.G2:.1f} & {E.ci(x.dif, x.dif_ic_inf, x.dif_ic_sup)} & "
                   f"{E.ci(x.dif_mesma_faixa_score, x.dif_mesma_faixa_score_ic_inf, x.dif_mesma_faixa_score_ic_sup)} & {_pf(x.get('p_mesma_faixa_score'))} \\\\")
    p = r["participacao"]
    return ("\\begin{table}[htbp]\n\\centering\\scriptsize\n"
            f"\\caption{{Default rates (\\%, 24-month client default indicator) by group. G1 = {rot['G1_en']}; "
            f"G2 = {rot['G2_en']}. ``Same risk class'' compares borrowers within the same credit-risk class "
            "(risk-class fixed effects). Contract level: 95\\% CIs clustered by client. Client level: one record per "
            f"client (group and score of the client's first contract in the subsample). Only "
            f"{p['pct_contratos_acima_corte']:.1f}\\% of the contracts ({p['pct_G1_acima_corte']:.1f}\\% of G1 and "
            f"{p['pct_G2_acima_corte']:.1f}\\% of G2) have a score above {r['corte']}. {E._desc_en(o.descricao())}.}}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{llrrrrccc}\n\\toprule\n"
            "Subsample & Level & $n$ G1 & $n$ G2 & Default G1 & Default G2 & G1 $-$ G2 (p.p.) & Same risk class (p.p.) & $p$ \\\\\n\\midrule\n"
            + "\n".join(lin) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def tab_inad_score_tex(r: dict, o: Opcoes, nivel: str = "contrato") -> str:
    f = r["por_faixa"]
    f = f[f.nivel == nivel]
    rot = _rot(o)
    lin = [f"{x.faixa_score} & {E.n(x.n_G1)} ({x.pct_dos_G1:.1f}\\%) & {E.n(x.n_G2)} ({x.pct_dos_G2:.1f}\\%) & "
           f"{x.G1:.1f} & {x.G2:.1f} & {E.ci(x.dif, x.dif_ic_inf, x.dif_ic_sup)} & {_pf(x.get('p_holm_ate_corte'))} \\\\"
           for _, x in f.iterrows()]
    return ("\\begin{table}[htbp]\n\\centering\\scriptsize\n"
            f"\\caption{{Default rates (\\%, 24-month client default indicator) by credit-risk class and group, "
            f"{'contract' if nivel == 'contrato' else 'client'} level. G1 = {rot['G1_en']}; G2 = {rot['G2_en']}. "
            "In parentheses: share of the group's "
            f"{'contracts' if nivel == 'contrato' else 'clients'} in the bin. 95\\% CIs "
            "clustered by client; $p$-values of the difference adjusted for multiple comparisons (Holm) across the bins with "
            f"scores up to {r['corte']}.}}\n"
            "\\begin{tabular}{lrrrrcc}\n\\toprule\n"
            f"Risk class (score) & $n$ G1 & $n$ G2 & Default G1 & Default G2 & G1 $-$ G2 (p.p.) & $p$ (Holm, score $\\leq$ {r['corte']}) \\\\\n\\midrule\n"
            + "\n".join(lin) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def fig_inad(r: dict, o: Opcoes, path: Path, nivel="contrato"):
    f = r["por_faixa"]
    f = f[f.nivel == nivel].reset_index(drop=True)
    rot = _rot(o)
    corte = r["corte"]
    x = np.arange(len(f))
    rights = [float(s.split(",")[1].strip(" ]")) for s in f.faixa_score]
    k = sum(rr <= corte + 1e-9 for rr in rights)
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    if k < len(f):
        ax.axvspan(k - 0.5, len(f) - 0.5, color="#999", alpha=0.12, lw=0)
        p = r["participacao"]
        ax.text(k - 0.4, 0.30 * max(f.G1.max(), f.G2.max()), f"score > {corte}:\n{p['pct_G1_acima_corte']:.1f}% of G1 and\n"
                f"{p['pct_G2_acima_corte']:.1f}% of G2 contracts", fontsize=7, color="#444", va="bottom")
    for g, dx in (("G2", -0.12), ("G1", 0.12)):
        # IC do nivel de cada grupo: aproximacao binomial (descritiva)
        pr = f[g] / 100
        se = 100 * np.sqrt(pr * (1 - pr) / f[f"n_{g}"])
        ax.errorbar(x + dx, f[g], yerr=1.96 * se, fmt="o-", ms=3.5, lw=1.2, capsize=2, color=E.COR[g],
                    label=f"{rot[g + '_en']} ({g})")
    ymax = max(f.G1.max(), f.G2.max())
    for i, row in f.iterrows():
        ax.text(i, 0.02 * ymax, f"{row.n_G1:,}\n{row.n_G2:,}", ha="center", va="bottom", fontsize=5.5, color="#444")
    ax.text(-0.55, 0.02 * ymax, "n G1\nn G2", ha="right", va="bottom", fontsize=5.5, color="#444")
    ax.set_xticks(x, f.faixa_score, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_ylabel("Default rate within 24 months (%)")
    ax.set_xlabel("Credit-risk class (score range; higher = riskier)")
    ax.set_xlim(-1.3, len(f) - 0.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ (b, c)

def fig_custo(c: dict, path: Path, versoes=tuple(VERSOES)):
    cv = c["curvas"]
    faixas, rights, corte = c["faixas"], c["rights"], c["corte"]
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, len(versoes), figsize=(4.0 * len(versoes) + 1.2, 3.8), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, v in zip(axs, versoes):
        s = cv[cv.versao_pd == v]
        for i, (fx, rr) in enumerate(zip(faixas, rights)):
            z = s[s.faixa_score == fx]
            ax.plot(100 * z.alpha, z.custo_pct_receita_faixa, lw=1.1, color=cmap(i / max(1, len(faixas) - 1)),
                    ls="-" if rr <= corte + 1e-9 else "--", label=fx)
        z = s[s.faixa_score == f"score <= {corte}"]
        ax.plot(100 * z.alpha, z.custo_pct_receita_faixa, color="black", lw=2.4,
                label=f"All bins with score ≤ {corte}\n(% of total interest revenue)")
        ax.set_title(VERSOES[v], fontsize=8.5)
        ax.set_xlabel("Share of the interest-rate difference eliminated (%)")
        ax.set_xticks(range(0, 101, 20))
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Cost (% of expected interest revenue of the class)")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7,
               title="Credit-risk class", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def tab_custo_tex(c: dict, o: Opcoes) -> str:
    rs = c["resumo_alpha1"]
    base = rs[rs.versao_pd == "score"].set_index("faixa_score")
    vs = list(VERSOES)
    lin = []
    cv = c["curvas"]
    for fx in c["faixas"] + [f"score <= {c['corte']}", "todas"]:
        row = cv[(cv.alpha == 1.0) & (cv.faixa_score == fx)].set_index("versao_pd")
        nome = {"todas": "All bins (\\% of total)", f"score <= {c['corte']}": f"Score $\\leq$ {c['corte']} (\\% of total)"}.get(fx, fx)
        n1 = E.n(base.loc[fx, "contratos_G1_com_reducao"]) if fx in base.index else "--"
        red = f"{base.loc[fx, 'reducao_media_pp']:.1f}" if fx in base.index else "--"
        vals = " & ".join(f"{row.loc[v, 'custo_pct_receita_faixa']:.1f}" for v in vs)
        rs_ = " & ".join(f"{row.loc[v, 'custo_R$'] / 1e6:.2f}" for v in vs)
        lin.append(f"{nome} & {n1} & {red} & {vals} & {rs_} \\\\")
        if fx == c["faixas"][-1]:
            lin.append("\\midrule")
    rot = _rot(o)
    return ("\\begin{table}[htbp]\n\\centering\\scriptsize\n"
            f"\\caption{{Cost of eliminating 100\\% of the within-cell interest-rate difference for G1 ({rot['G1_en']}), "
            "by credit-risk class. G1 rates are lowered by the cell difference only in cells where G1 pays more. "
            "Cost = loss of expected interest revenue, (1 $-$ PD) $\\times$ interest of a monthly-installment loan "
            "(Price table), as \\% of the expected interest revenue of the class (or of the whole portfolio in the last rows) "
            "and in R\\$ million. PD = credit score, or observed 24-month default rate of the borrower's group in the "
            f"credit-risk class. {E._desc_en(o.descricao())}.}}\n"
            "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            " & & & \\multicolumn{2}{c}{Cost (\\% of revenue)} & \\multicolumn{2}{c}{Cost (R\\$ million)} \\\\\n"
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "Risk class (score) & \\makecell{G1 contracts\\\\with reduction} & \\makecell{Mean reduction\\\\(p.p.)} & PD = score & PD observed "
            "& PD = score & PD observed \\\\\n\\midrule\n"
            + "\n".join(lin) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


# ------------------------------------------------------------------ gravacao

def exportar_etapa3(o: Opcoes, pasta: Path) -> dict:
    pasta.mkdir(parents=True, exist_ok=True)
    tag = o.tag()
    a = inadimplencia(o)
    c = custo(o)
    a["agregado"].to_csv(pasta / f"inadimplencia_agregado__{tag}.csv", index=False)
    a["por_faixa"].to_csv(pasta / f"inadimplencia_por_score__{tag}.csv", index=False)
    a["calibracao"].to_csv(pasta / f"calibracao_score__{tag}.csv", index=False)
    (pasta / f"tabela_inadimplencia__{tag}.tex").write_text(tab_inad_tex(a, o))
    (pasta / f"tabela_inadimplencia_por_score_contrato__{tag}.tex").write_text(tab_inad_score_tex(a, o, "contrato"))
    (pasta / f"tabela_inadimplencia_por_score_cliente__{tag}.tex").write_text(tab_inad_score_tex(a, o, "cliente"))
    fig_inad(a, o, pasta / f"fig_inadimplencia_por_score_contrato__{tag}.pdf", "contrato")
    fig_inad(a, o, pasta / f"fig_inadimplencia_por_score_cliente__{tag}.pdf", "cliente")
    c["curvas"].to_csv(pasta / f"custo_curvas__{tag}.csv", index=False)
    c["resumo_alpha1"].to_csv(pasta / f"custo_por_faixa_alpha100__{tag}.csv", index=False)
    (pasta / f"tabela_custo__{tag}.tex").write_text(tab_custo_tex(c, o))
    fig_custo(c, pasta / f"fig_custo_equidade__{tag}.pdf")
    return {"inadimplencia": a, "custo": c}


def gerar_etapa3(base: Path, progresso=None) -> Path:
    out = base / "etapa3"
    specs = [("P", Opcoes()), ("R1", Opcoes(amostra="completa")), ("R2", Opcoes(sem_renda="zero")),
             ("R3", Opcoes(amostra="completa", sem_renda="zero"))]
    linhas = []
    for i, (cod, o) in enumerate(specs):
        if progresso:
            progresso((i + 1) / len(specs), f"Etapa 3 {cod}")
        r = exportar_etapa3(o, out / o.tag())
        if cod == "P":
            exportar_etapa3(o, base)  # figuras e tabelas da analise principal tambem na raiz
        cv = r["custo"]["curvas"]
        for v in VERSOES:
            z = cv[(cv.versao_pd == v) & (cv.faixa_score == f"score <= {o.corte_score}")]
            linhas.append({"codigo": cod, "tag": o.tag(), "versao_pd": v,
                           **{f"custo_pct_alpha_{int(100 * a)}": z.loc[z.alpha == a, "custo_pct_receita_faixa"].iloc[0]
                              for a in ALPHAS if a > 0},
                           "custo_R$_alpha_100": z.loc[z.alpha == 1.0, "custo_R$"].iloc[0],
                           "receita_total_R$": z["receita_faixa_R$"].iloc[0]})
        for _, x in r["inadimplencia"]["agregado"].iterrows():
            linhas.append({"codigo": cod, "tag": o.tag(), "subamostra": x.subamostra, "nivel": x.nivel,
                           "G1": x.G1, "G2": x.G2, "dif": x.dif, "dif_ic_inf": x.dif_ic_inf, "dif_ic_sup": x.dif_ic_sup,
                           "dif_mesma_faixa_score": x.dif_mesma_faixa_score,
                           "dif_mesma_faixa_score_ic_inf": x.dif_mesma_faixa_score_ic_inf,
                           "dif_mesma_faixa_score_ic_sup": x.dif_mesma_faixa_score_ic_sup})
    pd.DataFrame(linhas).to_csv(out / "etapa3_principal_vs_robustez.csv", index=False)
    return out
