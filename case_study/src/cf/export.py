"""Exportacao: CSV, tabelas LaTeX (booktabs, em ingles para o artigo) e figuras PDF.

Cada combinacao de opcoes grava em outputs/<tag>/ e todos os nomes de arquivo
levam a tag, que identifica as opcoes (ver Opcoes.tag()).
So agregados sao gravados; nenhum registro individual sai de data/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.WARNING)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import ROOT, Opcoes, analisar, resumo_linha  # noqa: E402

OUT = ROOT / "outputs"
COR = {"G1": "#c0392b", "G2": "#2c6fbb"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def n(x, d=0):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


def ci(est, lo, hi, d=1):
    return f"{est:.{d}f} [{lo:.{d}f}; {hi:.{d}f}]"


def _desc_en(o: dict) -> str:
    amostra = {"janela": "sampling-window sample (contracts started July 2019 to June 2020)",
               "completa": "full sample (all personal-credit contracts in the data)"}[o["amostra"]]
    sr = {"excluir": "contracts without registered income excluded",
          "zero": "contracts without registered income kept with income = 0"}[o["sem_renda"]]
    med = {"renda": "loan-to-income ratio", "valor": "absolute loan amount", "nenhuma": None}[o["medida"]]
    lim = {"2SM": "two minimum wages in force at the contract start date",
           "2SM_cliente": "two minimum wages in force at the borrower's first contract"}.get(
        o["limite_renda"], f"R\\${float(o['limite_renda']) if o['limite_renda'] not in ('2SM', '2SM_cliente') else 0:,.0f}")
    if o["limite_renda"] == "2000":
        lim += " (about two minimum wages)"
    cel_score = "bank risk class" if o.get("faixas_score", "banco") == "banco" else "credit-risk class"
    ctrl = o["controles"]
    ctrl_en = {"nenhum": "none", "Numero de parcelas": "number of installments",
               "Default observado": "observed default", "Produto": "product",
               "Grupo de renda (limite escolhido)": "income group"}
    ctrl = ", ".join(ctrl_en.get(c.strip(), c) for c in ctrl.split(","))
    prod = f"; product {o['produto']} only" if o.get("produto") else ""
    celula = cel_score + ("" if med is None else f" $\\times$ {med} bin")
    return (f"{amostra}{prod}; {sr}; cells = {celula}"
            f"{'' if ctrl == 'none' else ' $\\times$ ' + ctrl}; income threshold: {lim}; "
            f"good-score cutoff: {o['corte_score']}")


# ------------------------------------------------------------------ tabelas

def tabela_resumo_tex(r: dict) -> str:
    rot = r["rotulos"]
    cols = [g for g in ("todos", "bom_score") if g in r["resultados"]]
    head = {"todos": "All scores", "bom_score": f"Score $\\leq$ {r['opcoes']['corte_score']}"}
    linhas = []

    def add(lab, f):
        linhas.append(lab + " & " + " & ".join(f(r["resultados"][g]) for g in cols) + r" \\")

    add("Contracts", lambda x: n(x["contratos"]))
    add("Clients", lambda x: n(x["clientes"]))
    add(f"Contracts, {rot['G1_en']} (G1)", lambda x: n(x["contratos_G1"]))
    add(f"Contracts, {rot['G2_en']} (G2)", lambda x: n(x["contratos_G2"]))
    add("Mean annual rate, G1 (\\%)", lambda x: n(x["taxa_media_G1"], 1))
    add("Mean annual rate, G2 (\\%)", lambda x: n(x["taxa_media_G2"], 1))
    linhas.append(r"\midrule")
    add("Raw difference G1 $-$ G2 (p.p.) [95\\% CI]",
        lambda x: ci(x["dif_bruta"]["est"], x["dif_bruta"]["ic_inf"], x["dif_bruta"]["ic_sup"]))
    add("Within-cell difference, regression (p.p.) [95\\% CI]",
        lambda x: ci(x["dif_celulas_reg"]["est"], x["dif_celulas_reg"]["ic_inf"], x["dif_celulas_reg"]["ic_sup"]))
    add("Fairness, Eq.~1 (p.p.) [95\\% CI]", lambda x: ci(x["eq1"], *x.get("eq1_ic", (np.nan, np.nan))))
    add("Fairness adjusted, Eq.~2 (p.p.) [95\\% CI]", lambda x: ci(x["eq2"], *x.get("eq2_ic", (np.nan, np.nan))))
    add("Cells with both groups", lambda x: n(x["celulas_validas"]))
    add("WDI, Eq.~3", lambda x: f"{x['wdi']:.2f}")
    add("WDI if no disparity, $(k-1)/(k+1)$", lambda x: f"{x['wdi_referencia_sem_disparidade']:.2f}")
    add("WDI with within-group shares", lambda x: f"{x['wdi_proporcoes']:.2f}")
    nota = (f"G1 = {rot['G1_en']}; G2 = {rot['G2_en']}. Positive differences: G1 pays more. "
            f"{_desc_en(r['opcoes'])}. Regression: annual rate on a G1 indicator with cell fixed "
            "effects; standard errors clustered by client. Eqs.~1--2: cells with both groups; "
            f"95\\% CI from a client-level bootstrap ({r['opcoes']['B']} replications).")
    return ("\\begin{table}[htbp]\n\\centering\\small\n"
            f"\\caption{{Interest-rate difference by {ATR_EN.get(r['opcoes']['atributo'], '')} group. {nota}}}\n"
            f"\\label{{tab:{r['tag'][:60]}}}\n"
            "\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n"
            " & " + " & ".join(head[g] for g in cols) + r" \\" + "\n\\midrule\n"
            + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def tabela_score_tex(r: dict) -> str:
    t = r["por_score"]
    rot = r["rotulos"]
    linhas = []
    for _, x in t.iterrows():
        dc = (f"{x.dif_reg:.1f} [{x.dif_reg_ic_inf:.1f}; {x.dif_reg_ic_sup:.1f}]"
              if "dif_reg" in t and not np.isnan(x.dif_reg) else "--")
        tt = "--" if "t" not in t or np.isnan(x.t) else f"{x.t:.1f}"
        pv = ("--" if "p_holm" not in t or np.isnan(x.p_holm)
              else ("$<$0.001" if x.p_holm < 0.001 else f"{x.p_holm:.3f}"))
        linhas.append(f"{x.faixa_score} & {n(x.n_G1)} & {n(x.n_G2)} & {n(x.taxa_G1, 1)} & "
                      f"{n(x.taxa_G2, 1)} & {n(x.dif_bruta, 1)} & {dc} & {tt} & {pv} & "
                      f"{n(100 * x.MAU_24_G1, 1)} & {n(100 * x.MAU_24_G2, 1)} \\\\")
    return ("\\begin{table}[htbp]\n\\centering\\scriptsize\n"
            f"\\caption{{Interest rates by credit-risk class and {ATR_EN.get(r['opcoes']['atributo'], 'group')} group. "
            f"G1 = {rot['G1_en']}; G2 = {rot['G2_en']}. IR = mean annual interest rate (\\%). Within-cell difference: "
            "regression of the annual rate on a G1 indicator with cell fixed effects inside the risk class; 95\\% CI and "
            "$t$ statistic with standard errors clustered by client; $p$-values adjusted for multiple comparisons "
            "across bins (Holm). Default = 24-month client default indicator. "
            f"{_desc_en(r['opcoes'])}.}}\n"
            "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n"
            "Risk class (score) & $n$ G1 & $n$ G2 & IR G1 & IR G2 & Raw diff. & Within-cell diff. [95\\% CI] & $t$ & $p$ (Holm) "
            "& Default G1 (\\%) & Default G2 (\\%) \\\\\n\\midrule\n"
            + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


ATR_EN = {"renda": "income", "sexo": "sex", "idade": "age", "escolaridade": "schooling",
          "estado_civil": "marital-status", "regiao": "location", "renda_cep": "postcode-income"}


def tabela_comparativa_tex(df: pd.DataFrame, titulo: str, rotulo: str, nomes: list[str]) -> str:
    linhas_def = [
        ("Contracts", lambda x: n(x["contratos"])),
        ("Clients", lambda x: n(x["clientes"])),
        ("Contracts G1 / G2", lambda x: f"{n(x['contratos_G1'])} / {n(x['contratos_G2'])}"),
        ("Mean rate G1 / G2 (\\%)", lambda x: f"{x['taxa_G1']:.1f} / {x['taxa_G2']:.1f}"),
        ("Raw difference (p.p.)", lambda x: ci(x["dif_bruta"], x["dif_bruta_ic_inf"], x["dif_bruta_ic_sup"])),
        ("Within-cell, regression (p.p.)", lambda x: ci(x["dif_celulas_reg"], x["dif_celulas_reg_ic_inf"], x["dif_celulas_reg_ic_sup"])),
        ("Fairness, Eq.~1 (p.p.)", lambda x: ci(x["eq1"], x.get("eq1_ic_inf", np.nan), x.get("eq1_ic_sup", np.nan))),
        ("Fairness adjusted, Eq.~2 (p.p.)", lambda x: ci(x["eq2"], x.get("eq2_ic_inf", np.nan), x.get("eq2_ic_sup", np.nan))),
        ("WDI (no-disparity reference)", lambda x: f"{x['wdi']:.2f} ({x['wdi_referencia']:.2f})"),
    ]
    body = []
    for lab, f in linhas_def:
        body.append(lab + " & " + " & ".join(f(x) for _, x in df.iterrows()) + r" \\")
    return ("\\begin{table}[htbp]\n\\centering\\scriptsize\n"
            f"\\caption{{{titulo}}}\n\\label{{tab:{rotulo}}}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{l" + "c" * len(df) + "}\n\\toprule\n"
            " & " + " & ".join(nomes) + r" \\" + "\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


# ------------------------------------------------------------------ figuras

def fig_taxa_por_score(r: dict, path: Path):
    t = r["por_score"]
    rot = r["rotulos"]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for g, dx in (("G2", -0.12), ("G1", 0.12)):
        se = t[f"dp_{g}"] / np.sqrt(t[f"n_{g}"])
        ax.errorbar(x + dx, t[f"taxa_{g}"], yerr=1.96 * se, fmt="o-", ms=3.5, lw=1.2, capsize=2,
                    color=COR[g], label=f"{rot[g + '_en']} ({g})")
    ax.set_xticks(x, t["faixa_score"], rotation=45, ha="right")
    ax.set_xlabel("Credit-risk class (score range; higher = riskier)")
    ax.set_ylabel("Mean annual interest rate (%)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_dif_por_score(r: dict, path: Path):
    """Diferenca de taxa G1 - G2 em cada faixa de score: (a) simples, todos os emprestimos da faixa;
    (b) comparando so emprestimos de tamanho parecido em relacao a renda (mesma celula), com IC 95%."""
    t = r["por_score"]
    rot = r["rotulos"]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.axhline(0, color="#555", lw=0.8)
    ax.plot(x, t["dif_bruta"], "s--", ms=3.5, lw=1, color="#888",
            label="All loans in the score bin (simple difference of means)")
    col = "dif_reg" if "dif_reg" in t else "dif_celulas_eq2"
    lo, hi = (("dif_reg_ic_inf", "dif_reg_ic_sup") if col == "dif_reg" else ("dif_celulas_ic_inf", "dif_celulas_ic_sup"))
    if lo in t:
        ax.fill_between(x, t[lo], t[hi], color=COR["G1"], alpha=0.15, lw=0)
    med = {"renda": "loan-to-income ratio", "valor": "loan amount", "nenhuma": None}.get(r["opcoes"]["medida"])
    lab = ("Loans of similar size relative to income (same risk class and loan-to-income bin), 95% CI"
           if med == "loan-to-income ratio" else
           "Same risk class and same loan-amount bin, 95% CI" if med else "Same risk class, 95% CI")
    ax.plot(x, t[col], "o-", ms=3.5, lw=1.4, color=COR["G1"], label=lab)
    corte = r["opcoes"]["corte_score"]
    rights = [float(s_.split(",")[1].strip(" ]")) for s_ in t["faixa_score"]]
    k = sum(rr <= corte + 1e-9 for rr in rights)
    if 0 < k < len(t):
        ax.axvline(k - 0.5, color="#555", ls=":", lw=1)
        ax.text(k - 0.45, ax.get_ylim()[1] * 0.92, f"score cutoff {corte}", fontsize=8, color="#555")
    ax.set_xticks(x, t["faixa_score"], rotation=45, ha="right")
    ax.set_xlabel("Credit-risk class (score range; higher = riskier)")
    ax.set_ylabel("Difference in annual rate (p.p.)")
    if r["opcoes"]["atributo"] == "renda":
        tit = "Lower-income (income below R$2,000) minus higher-income borrowers; positive = lower-income pay more"
    else:
        tit = f"{rot['G1_en']} (G1) minus {rot['G2_en']} (G2); positive = G1 pays more"
    ax.set_title(tit, fontsize=8)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_comparativa(df: pd.DataFrame, nomes: list[str], path: Path, titulo: str = ""):
    fig, ax = plt.subplots(figsize=(6.5, 0.5 + 0.55 * len(df)))
    y = np.arange(len(df))[::-1]
    sty = [("dif_bruta", "Raw", "#888", "s", -0.2), ("dif_celulas_reg", "Within-cell (regression)", COR["G1"], "o", 0.0),
           ("eq2", "Eq. 2", "#7d3c98", "D", 0.2)]
    for k, lab, c, m, dy in sty:
        lo = df.get(f"{k}_ic_inf", df[k]); hi = df.get(f"{k}_ic_sup", df[k])
        ax.errorbar(df[k], y + dy, xerr=[df[k] - lo, hi - df[k]], fmt=m, color=c, ms=4, capsize=2, label=lab)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_yticks(y, nomes)
    ax.set_xlabel("G1 $-$ G2 annual interest rate (percentage points, 95% CI)")
    if titulo:
        ax.set_title(titulo, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ gravacao

def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray, tuple)):
        return list(x)
    if isinstance(x, pd.DataFrame):
        return x.to_dict("records")
    return str(x)


def exportar(r: dict, base: Path | None = None) -> Path:
    tag = r["tag"]
    d = (base or OUT) / tag
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([resumo_linha(r, g) for g in r["resultados"]]).to_csv(d / f"resumo__{tag}.csv", index=False)
    r["cadeia"].to_csv(d / f"contagens__{tag}.csv", index=False)
    r["por_score"].to_csv(d / f"por_score__{tag}.csv", index=False)
    (d / f"tabela_resumo__{tag}.tex").write_text(tabela_resumo_tex(r))
    (d / f"tabela_por_score__{tag}.tex").write_text(tabela_score_tex(r))
    fig_taxa_por_score(r, d / f"fig_taxa_por_score__{tag}.pdf")
    fig_dif_por_score(r, d / f"fig_diferenca_por_score__{tag}.pdf")
    (d / f"resultados__{tag}.json").write_text(json.dumps(
        {k: v for k, v in r.items()}, default=_json_default, indent=1, ensure_ascii=False))
    return d


# ------------------------------------------------------------------ conjunto do artigo

PRINCIPAL = Opcoes()
CONJUNTO = {
    # (nome curto, rotulo em ingles para as tabelas, opcoes)
    "principal": [
        ("P", "Main: window,\\\\ income $>$ 0", PRINCIPAL),
    ],
    "robustez": [
        ("R1", "Full sample,\\\\ income $>$ 0", Opcoes(amostra="completa")),
        ("R2", "Window,\\\\ missing income = 0", Opcoes(sem_renda="zero")),
        ("R3", "Full sample,\\\\ missing income = 0", Opcoes(amostra="completa", sem_renda="zero")),
    ],
    "limite_renda": [
        ("L1", "2 minimum wages\\\\ at contract date", Opcoes(limite_renda="2SM")),
        ("L2", "2 minimum wages\\\\ at first contract", Opcoes(limite_renda="2SM_cliente")),
        ("L3", "Threshold\\\\ R\\$1,996", Opcoes(limite_renda="1996")),
        ("L1c", "Full sample, 2 MW\\\\ at contract date", Opcoes(amostra="completa", limite_renda="2SM")),
    ],
    "outros_atributos": [
        (f"A_{a}", lab, Opcoes(atributo=a)) for a, lab in
        (("sexo", "Women vs.\\\\ men"), ("idade", "Aged 60+ vs.\\\\ 18--59"),
         ("escolaridade", "Less than HS\\\\ vs. HS+"), ("estado_civil", "No partner vs.\\\\ partner"))
    ],
}


def _cel(s: str) -> str:
    return "\\makecell{" + s + "}"


def gerar_conjunto(progresso=None) -> Path:
    base = OUT / "conjunto_artigo"
    base.mkdir(parents=True, exist_ok=True)
    linhas = []
    total = sum(len(v) for v in CONJUNTO.values())
    k = 0
    resultados = {}
    for bloco, specs in CONJUNTO.items():
        for cod, lab, o in specs:
            k += 1
            if progresso:
                progresso(k / total, f"{cod}: {o.tag()}")
            r = analisar(o)
            exportar(r, base / "especificacoes")
            resultados[cod] = r
            for g in r["resultados"]:
                linhas.append({"bloco": bloco, "codigo": cod, "rotulo": lab.replace("\\\\", ""), **resumo_linha(r, g)})
    tab = pd.DataFrame(linhas)
    tab.to_csv(base / "conjunto_artigo_todas_especificacoes.csv", index=False)

    def comp(codigos, nome, titulo, grupo="todos"):
        sub = tab[(tab.codigo.isin(codigos)) & (tab.subamostra == grupo)].set_index("codigo").loc[codigos].reset_index()
        rot = {c: l for b in CONJUNTO.values() for c, l, _ in b}
        nomes = [_cel(rot[c]) for c in codigos]
        sufixo = "" if grupo == "todos" else f"_score_ate_{PRINCIPAL.corte_score:.2f}"
        (base / f"{nome}{sufixo}.tex").write_text(tabela_comparativa_tex(sub, titulo, nome + sufixo, nomes))
        sub.to_csv(base / f"{nome}{sufixo}.csv", index=False)
        fig_comparativa(sub, [rot[c].replace("\\\\", "").replace("$>$", ">").replace("\\$", "$").replace("--", "-") for c in codigos],
                        base / f"fig_{nome}{sufixo}.pdf")

    for g in ("todos", "bom_score"):
        comp(["P", "R1", "R2", "R3"], "tabela_principal_vs_robustez",
             "Income groups: main analysis (sampling window, contracts with registered income) and robustness "
             "(full sample; contracts without registered income kept with income = 0). Cells = credit-risk class "
             "$\\times$ loan-to-income bin; income threshold R\\$2,000 (about two minimum wages). "
             "G1 = income below the threshold. Positive: G1 pays more. 95\\% CIs clustered by client "
             "(regressions) or client bootstrap (Eqs.~1--2).", g)
        comp(["P", "L1", "L2", "L3", "R1", "L1c"], "tabela_limite_renda",
             "Sensitivity to the income threshold: fixed R\\$2,000 (main analysis) versus two minimum wages in force at the "
             "contract date (R\\$1,996 in 2019, R\\$2,078 in January 2020 and R\\$2,090 from February 2020), two minimum wages "
             "in force at the borrower's first contract, and a fixed R\\$1,996.", g)
        comp(["P", "A_sexo", "A_idade", "A_escolaridade", "A_estado_civil"], "tabela_atributos",
             "Other sensitive attributes, main sample and cells (credit-risk class $\\times$ loan-to-income bin). "
             "G1 is listed first in each column. Positive: G1 pays more.", g)
    exportar(resultados["P"], base)  # tabelas/figuras da analise principal tambem na raiz
    (base / "LEIA-ME.txt").write_text(
        "Conjunto do artigo gerado por src/cf/export.py:gerar_conjunto().\n"
        "P = analise principal; R1-R3 = robustez; L* = limite de renda; A_* = outros atributos.\n"
        "Subpasta especificacoes/: uma pasta por combinacao de opcoes (nome = opcoes).\n"
        "Arquivos *_score_ate_0.40.* = so contratos com score <= corte.\n")
    return base
