"""Correcoes de rotulo nas tabelas do artigo, aplicadas depois de copiar_para_paper.py (o programa esta congelado).

    python3 scripts/correcoes_tabelas.py

1. Servidor publico (Tabelas 2 e 3): o programa trata FUNC_PUB_ATIVIDADE == 5 como servidor, mas no codigo de
   2023 o codigo 5 corresponde aos nao servidores (setor privado); os codigos 1, 3 e 4 sao servidores municipais,
   estaduais ou federais. Aqui os rotulos da Tabela 3 sao trocados e a coluna "Public job" da Tabela 2 e
   recalculada como 1 menos o indicador do programa.
2. Tabela A4: por decisao da autora (24/11), o artigo nao separa os produtos. Saem o painel (b), com a diferenca
   dentro de cada produto, e as linhas do painel (c) com o produto na celula; fica a mistura de produtos de cada
   grupo (painel a) e a comparacao por valor absoluto.
3. Tabela 2: algumas rendas cadastradas sao quase nulas (minimo R$ 0,01); na coluna Loan/Income ficam so os quartis
   e a legenda explica.
4. Todas as tabelas: "R$" sem barra (vindo dos rotulos dos grupos) abre modo matematico no LaTeX; vira "R\\$".
O script e idempotente: nao reaplica correcoes ja feitas.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "paper" / "tables"
MARCA = "% correcao 1 (servidor publico) aplicada\n"

# ---------------------------------------------------------------- Tabela 3
f3 = T / "tab3_by_characteristic.tex"
s = f3.read_text()
if MARCA not in s:
    linhas = s.split("\n")
    i = next(k for k, l in enumerate(linhas) if "JOB: PUBLIC SERVANT" in l)
    no, yes = linhas[i + 1], linhas[i + 2]
    if no.startswith("No &") and yes.startswith("Yes &"):
        # o "Yes" do programa (codigo 5) vira "No"; o "No" do programa (codigos 1, 3 e 4) vira "Yes"
        linhas[i + 1] = "No &" + yes[len("Yes &"):]
        linhas[i + 2] = "Yes &" + no[len("No &"):]
    else:  # so uma linha (ex.: dados sinteticos sem essa variavel): inverte o rotulo dela
        trocar = {"No &": "Yes &", "Yes &": "No &"}
        for ini, fim in trocar.items():
            if no.startswith(ini):
                linhas[i + 1] = fim + no[len(ini):]
                break
    s = "\n".join(linhas)
    a = "IR = annual interest rate (\\%).}"
    assert s.count(a) == 1
    s = s.replace(a, "IR = annual interest rate (\\%). Public servant = municipal, state or federal government employee.}")
    f3.write_text(MARCA + s)

# ---------------------------------------------------------------- Tabela 2
f2 = T / "tab2_descriptives.tex"
s = f2.read_text()
if MARCA not in s:
    t2 = pd.read_csv(ROOT / "outputs" / "conjunto_artigo" / "artigo_descritivas" / "tabela2_descritivas.csv", index_col=0)
    pj = t2.loc["Public job"]
    novo = {"Mean": 1 - pj["mean"], "Std": pj["std"], "Min": 0.0, "25\\%": 1 - pj["75%"], "50\\%": 1 - pj["50%"],
            "75\\%": 1 - pj["25%"], "Max": 1.0}
    cab = next(l for l in s.split("\n") if l.startswith("\\textbf{Stat}"))
    cols = [c.strip() for c in cab.rstrip("\\ ").split("&")]
    j = next(k for k, c in enumerate(cols) if "Public job" in c)
    out = []
    for l in s.split("\n"):
        for rot, val in novo.items():
            if l.startswith(f"\\textbf{{{rot}}} &"):
                partes = l.rstrip()[:-2].rstrip().split("&") if l.rstrip().endswith("\\\\") else l.split("&")
                partes[j] = f" {val:.2f} "
                l = "&".join(partes).rstrip() + " \\\\"
        out.append(l)
    s = "\n".join(out)
    a = "Public job = government employee;"
    assert s.count(a) == 1
    s = s.replace(a, "Public job = public servant (municipal, state or federal government employee);")
    f2.write_text(MARCA + s)
# ---------------------------------------------------------------- Tabela A4
MARCA2 = "% correcao 2 (sem comparacao dentro de produto) aplicada\n"
fa = T / "tabA4_amount_product.tex"
s = fa.read_text()
if MARCA2 not in s:
    i = s.index("\\multicolumn{7}{l}{\\textit{(b) Difference")
    j = s.index("\\multicolumn{7}{l}{\\textit{(c) Loan size")
    s = s[:i] + s[j:]
    s = s.replace("\\textit{(c) Loan size measured in reais, and product controls (p.p.)}", "\\textit{(b) Loan size measured in reais (p.p.)}")
    s = "\n".join(l for l in s.split("\n") if "+ product &" not in l)
    a = "product mix, differences within products and with absolute loan amounts."
    assert s.count(a) == 1
    s = s.replace(a, "product mix and comparison with absolute loan amounts.")
    fa.write_text(MARCA2 + s)
# ---------------------------------------------------------------- Tabela 2: Loan/Income so quartis
MARCA3 = "% correcao 3 (Loan/Income so quartis) aplicada\n"
f2 = T / "tab2_descriptives.tex"
s = f2.read_text()
if MARCA3 not in s:
    cab = next(l for l in s.split("\n") if l.startswith("\\textbf{Stat}"))
    cols = [c.strip() for c in cab.rstrip("\\ ").split("&")]
    j = next(k for k, c in enumerate(cols) if "Loan/Income" in c)
    out = []
    for l in s.split("\n"):
        if any(l.startswith(f"\\textbf{{{r}}} &") for r in ("Mean", "Std", "Min", "Max")):
            partes = l.rstrip()[:-2].rstrip().split("&")
            partes[j] = " -- "
            l = "&".join(partes).rstrip() + " \\\\"
        out.append(l)
    s = "\n".join(out)
    a = "Income = registered monthly income (R\\$);"
    assert s.count(a) == 1, "legenda tab2"
    s = s.replace(a, a + " a few registered incomes are close to zero (minimum R\\$0.01), which makes the mean and the "
                  "extremes of the loan-to-income ratio meaningless, so only its quartiles are shown; these contracts are kept "
                  "and fall in the highest loan-to-income bin;")
    f2.write_text(MARCA3 + s)
# ---------------------------------------------------------------- todas: R$ sem barra
import re  # noqa: E402
for f in sorted(T.glob("*.tex")):
    s = f.read_text()
    s2 = re.sub(r"(?<!\\)R\$", lambda m: "R\\$", s)
    if s2 != s:
        f.write_text(s2)
print("correcoes aplicadas em", T)
