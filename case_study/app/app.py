"""Interface local para escolher as opcoes de analise e gerar tabelas e figuras.

    cd ~/Desktop/UNIRIO_2026/files/Discriminacao/CostFairness_revisao_2026-11-23
    streamlit run app/app.py

Abre em http://localhost:8766 (so nesta maquina; ver .streamlit/config.toml).
Le a base confidencial em data/derived/ e grava so agregados em outputs/.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.cf import analysis as A  # noqa: E402
from src.cf import export as E  # noqa: E402

st.set_page_config(page_title="Cost of Fairness - analise", layout="wide")
st.title("The Cost of Fairness: escolha das opcoes de analise")

if not A.BASE.exists():
    st.error("Base nao encontrada. Rode antes: `python3 -m src.cf.build` na pasta do projeto.")
    st.stop()

# ------------------------------------------------------------------ opcoes
with st.sidebar:
    st.header("Opcoes")
    st.caption("Os valores iniciais sao os do artigo (Report24Nov): so ID, janela, sem os contratos sem renda, "
               "emprestimo/renda, sem controles adicionais, renda < R$ 2.000, classes de risco do banco, corte 0,4.")
    unidade = st.radio("Unidade de contrato", ["id", "id_valor"],
                       format_func={"id": "So ID, como no artigo (principal)",
                                    "id_valor": "ID + valor inicial (alternativa)"}.get)
    amostra = st.radio("Amostra", ["janela", "completa"],
                       format_func={"janela": "Janela jul/2019-jun/2020 (principal)",
                                    "completa": "Completa (todos os contratos)"}.get)
    sem_renda = st.radio("Contratos sem renda", ["excluir", "zero"],
                         format_func={"excluir": "Excluir (principal)", "zero": "Tratar como renda zero"}.get)
    medida = st.radio("Medida de tamanho do emprestimo", ["renda", "valor"],
                      format_func={"renda": "Emprestimo / renda (principal)", "valor": "Valor absoluto"}.get)
    faixas_score = st.radio("Faixas de score", ["banco", "antigas"],
                            format_func={"banco": "12 classes de risco do banco (artigo)",
                                         "antigas": "Faixas antigas (22 na celula, 13 na tabela)"}.get)
    controles = st.multiselect("Controles adicionais (entram na celula)", list(A.CONTROLES),
                               format_func=A.CONTROLES.get)
    atributo = st.selectbox("Atributo sensivel", list(A.ATR), format_func=lambda a: A.ATR[a]["rotulo"])
    lim_tipo = st.selectbox("Limite de renda", ["2000", "2SM", "2SM_cliente", "1996", "outro"],
                            format_func={"2000": "R$ 2.000 fixo (artigo)",
                                         "2SM": "2 salarios minimos na data do contrato",
                                         "2SM_cliente": "2 SM na data do 1o contrato do cliente",
                                         "1996": "R$ 1.996 fixo (2 SM de 2019)", "outro": "Outro valor"}.get)
    limite = lim_tipo
    if lim_tipo == "outro":
        limite = str(st.number_input("Limite (R$)", min_value=100.0, value=2000.0, step=50.0))
    incl = st.radio("Quem fica exatamente no limite", [False, True],
                    format_func={False: "No grupo de maior renda: G1 = renda < limite (artigo)", True: "No grupo de menor renda: G1 = renda <= limite"}.get)
    corte = st.number_input("Score de corte (bom score = score <= corte)", 0.05, 1.0, 0.4, 0.05)
    with st.expander("Avancado"):
        min_n = st.number_input("Minimo de contratos de cada grupo na celula (artigo: 1)", 1, 100, 1)
        B = st.number_input("Replicas do bootstrap", 0, 2000, 200, 50)

o = A.Opcoes(unidade=unidade, faixas_score=faixas_score, amostra=amostra, sem_renda=sem_renda, medida=medida, controles=tuple(sorted(controles)),
             atributo=atributo, limite_renda=limite, limite_inclusivo=incl, corte_score=float(corte),
             min_n=int(min_n), B=int(B))


@st.cache_data(show_spinner="Calculando...", max_entries=40)
def rodar(o: A.Opcoes):
    return A.analisar(o)


r = rodar(o)
st.caption(f"Identificador das opcoes (usado nos nomes de arquivo): `{r['tag']}`")
rot = {k: (v.replace('$', '\\$') if isinstance(v, str) else v) for k, v in r["rotulos"].items()}

# ------------------------------------------------------------------ contagens
c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Contagem de contratos e clientes")
    st.dataframe(r["cadeia"].style.format({"contratos": "{:,}", "clientes": "{:,}"}), hide_index=True,
                 use_container_width=True)
with c2:
    k = r["contagens"]
    st.subheader("Grupos")
    st.markdown(f"**G1** = {rot['G1']}: {k['contratos_G1']:,} contratos, {k['clientes_G1']:,} clientes  \n"
                f"**G2** = {rot['G2']}: {k['contratos_G2']:,} contratos, {k['clientes_G2']:,} clientes")
    if atributo == "renda":
        st.markdown(f"Limite aplicado: R\\$ {k['limite_renda_min']:,.0f} a R\\$ {k['limite_renda_max']:,.0f}"
                    + (f"; {k['contratos_renda_zero']:,} contratos com renda = 0" if sem_renda == "zero" else ""))
    st.markdown("Diferenca = taxa(G1) - taxa(G2), em p.p. ao ano; **positiva = G1 paga mais**.")

# ------------------------------------------------------------------ resultados
st.subheader("Diferenca de taxa e metricas de fairness")
lin = []
for g, nome in (("todos", "Todos os scores"), ("bom_score", f"Score <= {corte}")):
    x = r["resultados"].get(g)
    if not x:
        continue
    f = lambda v, lo, hi: f"{v:.2f}  [{lo:.2f}; {hi:.2f}]"  # noqa: E731
    lin.append({
        "Subamostra": nome, "Contratos": f"{x['contratos']:,}", "Clientes": f"{x['clientes']:,}",
        "Taxa G1 (%)": f"{x['taxa_media_G1']:.1f}", "Taxa G2 (%)": f"{x['taxa_media_G2']:.1f}",
        "Dif. bruta [IC95]": f(x["dif_bruta"]["est"], x["dif_bruta"]["ic_inf"], x["dif_bruta"]["ic_sup"]),
        "Dif. na celula, regressao [IC95]": f(x["dif_celulas_reg"]["est"], x["dif_celulas_reg"]["ic_inf"],
                                             x["dif_celulas_reg"]["ic_sup"]),
        "Eq. 1 direta [IC95]": f(x["eq1"], *x.get("eq1_ic", (float("nan"),) * 2)),
        "Eq. 2 ajustada [IC95]": f(x["eq2"], *x.get("eq2_ic", (float("nan"),) * 2)),
        "Celulas validas": f"{x['celulas_validas']:,} de {x['celulas_total']:,}",
        "WDI (Eq. 3)": f"{x['wdi']:.3f}",
        "WDI sem disparidade (k-1)/(k+1)": f"{x['wdi_referencia_sem_disparidade']:.3f}",
        "WDI com proporcoes": f"{x['wdi_proporcoes']:.3f}",
    })
st.dataframe(pd.DataFrame(lin).set_index("Subamostra").T, use_container_width=True)
st.caption("IC da diferenca bruta e da regressao: erros agrupados por cliente. IC das Eqs. 1 e 2: "
           f"bootstrap por cliente ({o.B} replicas). Celula = faixa de score x faixa de "
           f"{'emprestimo/renda' if medida == 'renda' else 'valor absoluto'}"
           + (" x " + " x ".join(A.CONTROLES[c].lower() for c in o.controles) if o.controles else "") + ".")

st.subheader("Por faixa de score")
t = r["por_score"].copy()
st.dataframe(t.style.format({c: "{:.2f}" for c in t.columns if t[c].dtype.kind == "f"}), hide_index=True,
             use_container_width=True)

g1, g2 = st.columns(2)
tmp = ROOT / "outputs" / "_tmp_app"
tmp.mkdir(parents=True, exist_ok=True)
for col, fn, nome in ((g1, E.fig_taxa_por_score, "taxa"), (g2, E.fig_dif_por_score, "dif")):
    p = tmp / f"{nome}.png"
    fn(r, p)
    col.image(str(p))
plt.close("all")

# ------------------------------------------------------------------ exportar
st.divider()
e1, e2 = st.columns(2)
with e1:
    st.subheader("Exportar esta combinacao")
    if st.button("Gravar tabelas LaTeX, figuras PDF e CSV"):
        d = E.exportar(r)
        st.success(f"Gravado em `{d.relative_to(ROOT)}`")
        st.write(sorted(p.name for p in d.iterdir()))
with e2:
    st.subheader("Conjunto do artigo")
    st.markdown("Analise principal (janela, sem os contratos sem renda, emprestimo/renda, renda < R$ 2.000, classes de risco do banco, "
                "corte 0,4) + robustez (amostra completa; renda zero), sensibilidade do limite de renda "
                "e os outros atributos.")
    if st.button("Gerar conjunto do artigo (~8 min)", type="primary"):
        bar = st.progress(0.0, "iniciando")
        d = E.gerar_conjunto(lambda f, m: bar.progress(f, m[:90]))
        from src.cf.export3 import gerar_etapa3
        gerar_etapa3(d, lambda f, m: bar.progress(f, m[:90]))
        from src.cf.etapa4 import gerar_etapa4
        gerar_etapa4(d, lambda f, m: bar.progress(f, m[:90]))
        from src.cf import paper_tables, xgb
        bar.progress(0.95, "Tabelas 2-3, Fig. 7 e XGBoost")
        paper_tables.gerar(d)
        tx = xgb.tabela()
        tx.to_csv(d / "robustez_xgboost.csv", index=False)
        (d / "tabela_robustez_xgboost.tex").write_text(xgb.tabela_tex(tx))
        st.success(f"Gravado em `{d.relative_to(ROOT)}`")
        comp = pd.read_csv(d / "tabela_principal_vs_robustez.csv")
        st.dataframe(comp[["codigo", "contratos", "clientes", "dif_bruta", "dif_celulas_reg", "eq1", "eq2",
                           "wdi"]].round(2), hide_index=True)

# ------------------------------------------------------------------ Etapa 3
st.divider()
st.header("Etapa 3: inadimplencia e custo da equidade")
st.caption("Usa as opcoes da barra lateral (amostra, renda, atributo, celulas, corte de score).")
if st.checkbox("Calcular inadimplencia e custo para estas opcoes", value=True):
    from src.cf import default_cost as DC
    from src.cf import export3 as E3

    @st.cache_data(show_spinner="Calculando inadimplencia e custo...", max_entries=20)
    def etapa3(o):
        return DC.inadimplencia(o), DC.custo(o)

    a3, c3 = etapa3(o)
    st.subheader("Inadimplencia (%) de G1 e G2")
    st.dataframe(a3["agregado"].round(2), hide_index=True, use_container_width=True)
    st.caption("Default = indicador do cliente em 24 meses (MAU_24). dif = G1 - G2 em p.p., IC 95% com erros "
               "agrupados por cliente; dif_mesma_faixa_score = com efeito fixo de faixa de score.")
    pp = a3["participacao"]
    st.markdown(f"Score acima de {o.corte_score}: **{pp['pct_contratos_acima_corte']:.1f}%** dos contratos "
                f"({pp['pct_G1_acima_corte']:.1f}% dos de G1 e {pp['pct_G2_acima_corte']:.1f}% dos de G2).")
    nivel = st.radio("Nivel", ["contrato", "cliente"], horizontal=True)
    st.dataframe(a3["por_faixa"][a3["por_faixa"].nivel == nivel].round(2), hide_index=True,
                 use_container_width=True)
    p = tmp / "inad.png"
    E3.fig_inad(a3, o, p, nivel)
    st.image(str(p))
    st.subheader("Custo de reduzir a diferenca de taxas")
    vs = st.multiselect("Versoes do default", list(E3.VERSOES), default=list(E3.VERSOES),
                        format_func=E3.VERSOES.get)
    if vs:
        p = tmp / "custo.png"
        E3.fig_custo(c3, p, tuple(vs))
        st.image(str(p))
    cv = c3["curvas"]
    piv = cv[cv.faixa_score.isin([f"score <= {o.corte_score}", "todas"])].pivot_table(
        index="alpha", columns=["versao_pd", "faixa_score"], values="custo_pct_receita_faixa")
    st.markdown("Custo em % da receita de juros esperada da carteira")
    st.dataframe(piv.round(2), use_container_width=True)
    st.dataframe(c3["resumo_alpha1"].round(2), hide_index=True, use_container_width=True)
    if st.button("Gravar tabelas e figuras da Etapa 3"):
        d3 = ROOT / "outputs" / o.tag() / "etapa3"
        E3.exportar_etapa3(o, d3)
        st.success(f"Gravado em `{d3.relative_to(ROOT)}`")

# ------------------------------------------------------------------ Etapa 4
st.divider()
st.header("Etapa 4: com e sem controles")
st.caption("Escada de controles para o atributo escolhido na barra lateral, nas duas amostras. "
           "C0 sem controles; C1 mesma faixa de score; C2 celula score x emprestimo/renda; C3 C2 + parcelas; "
           "C4 C2 + grupo de renda; C5 C2 + parcelas + grupo de renda + produto. Leva ~1 min.")
if st.button("Calcular escada de controles"):
    from src.cf import etapa4 as E4
    with st.spinner("Calculando..."):
        esc = pd.concat([E4.escada(o.atributo, am, B=o.B) for am in ("janela", "completa")], ignore_index=True)
    esc["diferenca [IC95]"] = esc.apply(lambda x: f"{x.dif:+.2f} [{x.ic_inf:+.2f}; {x.ic_sup:+.2f}]", axis=1)
    st.dataframe(esc.pivot(index=["nivel", "controles"], columns="amostra", values="diferenca [IC95]"),
                 use_container_width=True)
    d4 = ROOT / "outputs" / f"etapa4__attr-{o.atributo}"
    d4.mkdir(parents=True, exist_ok=True)
    esc.to_csv(d4 / f"escada_controles__attr-{o.atributo}.csv", index=False)
    st.success(f"Gravado em `{d4.relative_to(ROOT)}`")

# ------------------------------------------------------------------ XGBoost
st.divider()
st.header("Robustez sem faixas: XGBoost")
st.caption("Treina o XGBoost (score e emprestimo/renda) em um grupo e aplica no outro; diferenca = taxa observada - "
           "prevista, sempre G1 - G2. Usa a amostra, a renda e o atributo da barra lateral. Leva ~30 s.")
if st.button("Calcular contrafactual XGBoost"):
    from src.cf import xgb
    with st.spinner("Treinando..."):
        rx = pd.DataFrame([xgb.contrafactual(o, treino=k, B=o.B) for k in (0, 1)])
    rx["params"] = rx["params"].astype(str)
    st.dataframe(rx.round(3), hide_index=True, use_container_width=True)
