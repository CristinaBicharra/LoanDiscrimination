"""Copia tabelas (.tex) e figuras (.pdf) de outputs/conjunto_artigo para paper/, com rotulos fixos.

    python3 scripts/copiar_para_paper.py
Depois: python3 scripts/montar_report_nov23.py (os numeros do texto estao escritos nesse script).
"""
import glob, re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
O = str(ROOT / "outputs" / "conjunto_artigo") + "/"
T = ROOT / "paper" / "tables"
F = ROOT / "paper" / "figures"
T.mkdir(parents=True, exist_ok=True)
F.mkdir(parents=True, exist_ok=True)
g = lambda pat: glob.glob(O + pat)[0]
tabs = {
    "tab2_descriptives": (O + "artigo_descritivas/tabela2_descritivas.tex", "tab:2"),
    "tab3_by_characteristic": (O + "artigo_descritivas/tabela3_por_caracteristica.tex", "tab:3"),
    "tab4_income_by_score_bin": (g("amostra-janela__*/tabela_por_score__*.tex"), "tab:4"),
    "tab5_income_summary": (g("amostra-janela__*/tabela_resumo__*.tex"), "tab:5"),
    "tab6_attributes": (O + "etapa4/tabela_escada_controles_janela.tex", "tab:6"),
    "tab7_default": (g("tabela_inadimplencia__*.tex"), "tab:7"),
    "tab7b_default_by_bin": (g("tabela_inadimplencia_por_score_contrato__*.tex"), "tab:7b"),
    "tab7b_default_by_bin": (g("tabela_inadimplencia_por_score_contrato__*.tex"), "tab:7b"),
    "tab8_cost": (g("tabela_custo__*.tex"), "tab:8"),
    "tabA1_robustness": (O + "tabela_principal_vs_robustez.tex", "tab:A1"),
    "tabA2_threshold": (O + "tabela_limite_renda.tex", "tab:A2"),
    "tabA3_xgboost": (O + "tabela_robustez_xgboost.tex", "tab:A3"),
    "tabA4_amount_product": (O + "etapa4/tabela_produto_valor_absoluto.tex", "tab:A4"),
}
for nome, (src, lab) in tabs.items():
    s = open(src).read()
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    # sem numeros fixos de equacao e sem travessao nas legendas
    s = s.replace("Eqs.~1--2", "Eqs.~\\ref{eq:1} and~\\ref{eq:2}").replace("Eq.~1", "Eq.~\\ref{eq:1}")
    s = s.replace("Eq.~2", "Eq.~\\ref{eq:2}").replace("Eq.~3", "Eq.~\\ref{eq:3}")
    s = s.replace("Jul 2019--Jun 2020", "July 2019 to June 2020").replace("C1--C5", "C1 to C5")
    for _a, _b in [('Credit-score bin (higher = riskier)', 'Credit-risk class (score range; higher = riskier)'), ('score-bin fixed effects', 'risk-class fixed effects'), ("``Same score bin''", "``Same risk class''"), ('Same score bin', 'Same risk class'), ('same score bin', 'same risk class'), ('inside the score bin', 'inside the risk class'), ('credit-score bin', 'credit-risk class'), ('Score bin &', 'Risk class (score) &'), ('Score bin', 'Risk class (score)'), ('title="Credit-score bin"', 'title="Credit-risk class"'), ('of the bin (or of the whole portfolio', 'of the class (or of the whole portfolio'), ('revenue of the bin)', 'revenue of the class)'), ("group in the score bin.", "group in the credit-risk class.")]:
        s = s.replace(_a, _b)
    s = s.replace("\\caption{\\add{", "\\caption{")
    if "longtable" in s:
        s = s.replace("{\\scriptsize", "{\\scriptsize\\ifshowchanges\\color{blue}\\fi", 1)
    else:
        s = s.replace("\\caption{", "\\ifshowchanges\\color{blue}\\fi\n\\caption{", 1)
    i = s.index("\\caption{"); k = i + len("\\caption"); depth = 0
    while True:
        ch = s[k]
        if ch == "\\":
            k += 2; continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    rest = s[k + 1:]
    if rest.startswith("}"):  # chave que fechava o \add removido
        rest = rest[1:]
    s = s[:k + 1] + "\\label{" + lab + "}" + rest
    if "longtable" not in s and "\\resizebox" not in s:
        s = s.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
        s = s.replace("\\end{tabular}", "\\end{tabular}}", 1)
    if nome == "tab8_cost":
        s = s.replace("& G1 contracts with reduction & Mean reduction (p.p.) &",
                      "& \\makecell{G1 contracts\\\\with reduction} & \\makecell{Mean reduction\\\\(p.p.)} &")
    if nome == "tabA4_amount_product":
        s = s.replace("\\begin{tabular}{lrrrrrr}", "\\begin{tabular}{p{5.2cm}rrrrrr}")
    (T / f"{nome}.tex").write_text("% gerado por src/cf (CostFairness_revisao_2026-11-23); nao editar a mao\n" + s)
figs = {"fig7_rates_by_score_attributes.pdf": O + "artigo_descritivas/fig7_taxa_por_score_atributos.pdf",
        "fig8_default_by_income.pdf": g("fig_inadimplencia_por_score_contrato__*.pdf"),
        "fig9_cost_of_fairness.pdf": g("fig_custo_equidade__*.pdf")}
for d, s in figs.items():
    shutil.copy(s, F / d)
print("copiado:", len(tabs), "tabelas e", len(figs), "figuras")
