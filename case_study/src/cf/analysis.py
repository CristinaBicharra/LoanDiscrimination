"""Motor de analise: uma combinacao de opcoes -> contagens, diferenca de taxa,
metricas de fairness do artigo (Eqs. 1-3).

Convencoes
* Taxa = taxa anual composta, ((1 + taxa mensal)^12 - 1) x 100, em % a.a.
* G1 = grupo potencialmente desfavorecido; G2 = referencia.
  Diferenca = taxa(G1) - taxa(G2), em pontos percentuais; positiva = G1 paga mais.
* Celula = faixa de score x faixa de tamanho do emprestimo (emprestimo/renda ou
  valor absoluto) [x controles adicionais escolhidos: faixa de parcelas,
  default observado, produto].
* Eq. 1 (direta): media simples das diferencas nas celulas que tem os dois
  grupos (com pelo menos `min_n` contratos em cada grupo).
* Eq. 2 (ajustada): media das mesmas diferencas ponderada por (n_G1 + n_G2).
* Intervalos das Eqs. 1-2: bootstrap por cliente (reamostragem de Poisson dos
  clientes, `B` replicas, percentis 2,5 e 97,5).
* Diferenca por regressao: taxa ~ G1 + efeitos fixos de celula, erros-padrao
  agrupados por cliente (CRV1). Sem celulas = diferenca bruta.
* Eq. 3 (WDI): faixas de taxa, contagens de G2 (dominante) e G1, peso e^-i
  (i = posicao da faixa, 1 = menor taxa), so faixas nao vazias. Tambem se
  reporta a referencia de "sem disparidade" (k-1)/(k+1), k = N_G2 / N_G1, e a
  versao com proporcoes dentro de cada grupo.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "config" / "opcoes.json").read_text())
from .build import caminho, settings

BASE = caminho("id_valor")
CLI = "CHAVE_ALEATORIA"
N_FAIXAS_VALOR = 22
PRODUTO_MIN = 500  # produtos com menos contratos (amostra completa) viram "OUTROS"


def _edges(lst):
    return [np.inf if x == "inf" else float(x) for x in lst]


FX = {k: _edges(v) for k, v in CFG["faixas"].items() if isinstance(v, list)}
ATR = CFG["atributos"]
CONTROLES = {"parcelas": "Numero de parcelas", "default": "Default observado",
             "faixa_renda": "Grupo de renda (limite escolhido)",
             "produto": "Produto"}


# ------------------------------------------------------------------ opcoes

@dataclass(frozen=True)
class Opcoes:
    amostra: str = "janela"            # "janela" | "completa"
    unidade: str = "id"                # "id" (codigo de 2023; principal, decisao 23/11) | "id_valor"
    sem_renda: str = "excluir"         # "excluir" | "zero"
    medida: str = "renda"              # "renda" (emprestimo/renda) | "valor" (absoluto)
    controles: tuple = ()              # subconjunto de CONTROLES
    atributo: str = "renda"            # renda | sexo | idade | escolaridade | estado_civil
    limite_renda: str = "2000"         # artigo: R$ 2.000 fixo; "2SM" = 2 SM na data do contrato; "2SM_cliente"; ou valor em R$
    limite_inclusivo: bool = False     # False (artigo, igual a versao enviada): G1 = renda < limite; True: <=
    corte_score: float = 0.4
    min_n: int = 1                     # minimo de contratos de cada grupo na celula
    produto: str = ""                  # se preenchido, so contratos deste produto (GRUPO_PRODUTO_CONTRATADO)
    max_parcelas: int = 0              # se > 0, so contratos com ate este numero de parcelas
    faixas_score: str = "banco"        # artigo: "banco" (12 classes de risco do banco); "antigas" (22 faixas na celula, 13 na tabela)
    B: int = 200                       # replicas do bootstrap
    seed: int = 20261123

    def tag(self) -> str:
        ctrl = "+".join(sorted(self.controles)) or "nenhum"
        lim = self.limite_renda if self.limite_renda in ("2SM", "2SM_cliente") else f"R{float(self.limite_renda):.0f}"
        partes = ([] if self.unidade == "id" else [f"unid-{self.unidade}"]) + [f"amostra-{self.amostra}", f"semrenda-{self.sem_renda}",
                  f"medida-{ {'renda': 'empr_renda', 'valor': 'valor', 'nenhuma': 'so_score'}[self.medida] }",
                  f"ctrl-{ctrl}", f"attr-{self.atributo}", f"lim-{lim}{'-incl' if self.limite_inclusivo else ''}",
                  f"corte-{self.corte_score:.2f}"]
        if self.min_n > 1:
            partes.append(f"minn-{self.min_n}")
        if self.faixas_score != "banco":
            partes.append(f"score-{self.faixas_score}")
        if self.max_parcelas:
            partes.append(f"parc-ate-{self.max_parcelas}")
        if self.produto:
            partes.append("prod-" + "".join(ch if ch.isalnum() else "_" for ch in self.produto)[:30])
        return "__".join(partes)

    def descricao(self) -> dict:
        d = asdict(self)
        d["controles"] = ", ".join(CONTROLES[c] for c in self.controles) or "nenhum"
        return d


# ------------------------------------------------------------------ dados

@lru_cache(maxsize=2)
def base(unidade: str = "id") -> pd.DataFrame:
    b = pd.read_parquet(caminho(unidade))
    b["MES_INICIO"] = b["DATA_INICIO_CONTRATO"].dt.to_period("M")
    prod = b["GRUPO_PRODUTO_CONTRATADO"]
    vc = prod.value_counts()
    b["PRODUTO"] = prod.where(prod.isin(vc[vc >= PRODUTO_MIN].index), "OUTROS")
    # localizacao (so o estado de origem do banco): fora x dentro da regiao metropolitana da capital;
    # renda media do CEP (Censo 2022). Parametros em config/private_settings.json (nao publicado).
    loc = settings().get("location", {})
    b["FORA_METROPOLITANA"] = np.nan
    b["CEP_RENDA_MEDIA"] = np.nan
    b["CEP_RENDA_BAIXA"] = np.nan
    arq = ROOT / loc.get("postcode_income_file", "_")
    if loc and "UF_RESIDENCIAL" in b and "CEP_RESIDENCIAL" in b:
        no_estado = b["UF_RESIDENCIAL"] == loc["state"]
        cep = b["CEP_RESIDENCIAL"].astype(str).str.zfill(8)
        pref = pd.to_numeric(cep.str[:2], errors="coerce")
        e0, e1 = loc["state_postcode_prefixes"]
        m0, m1 = loc["metro_postcode_prefixes"]
        b["FORA_METROPOLITANA"] = np.where(no_estado & pref.between(e0, e1),
                                           (~pref.between(m0, m1)).astype(float), np.nan)
        if arq.is_file():
            rc = pd.read_csv(arq, dtype={"CEP": str})
            b["CEP_RENDA_MEDIA"] = cep.map(rc.set_index("CEP")["Renda_media_reais"]).where(no_estado)
            b["CEP_RENDA_BAIXA"] = np.where(b["CEP_RENDA_MEDIA"].notna(),
                                            (b["CEP_RENDA_MEDIA"] < loc["postcode_income_median"]).astype(float), np.nan)
    return b


@lru_cache(maxsize=2)
def faixas_valor(unidade: str = "id") -> list:
    """22 faixas de valor absoluto: quantis da amostra completa (fixas para todas as opcoes)."""
    v = base(unidade)["VLR_CONTRATADO"].dropna()
    q = np.unique(np.round(v.quantile(np.linspace(0, 1, N_FAIXAS_VALOR + 1)).to_numpy(), 0))
    q[0], q[-1] = 0.0, np.inf
    return list(q)


def salario_minimo(mes: pd.Series) -> pd.Series:
    tab = sorted((pd.Period(k, "M"), v) for k, v in CFG["salario_minimo_mensal"].items()
                 if not k.startswith("_"))
    out = pd.Series(np.nan, index=mes.index)
    for p, v in tab:
        out[mes >= p] = v
    return out


def _regra(x: pd.Series, regra: str) -> pd.Series:
    regra = regra.strip()
    if regra.startswith("not in"):
        return ~x.isin(json.loads(regra[6:].strip()))
    op, v = regra.split()
    v = float(v)
    return {"<": x < v, "<=": x <= v, ">": x > v, ">=": x >= v, "==": x == v}[op]


def preparar(o: Opcoes) -> tuple[pd.DataFrame, list]:
    """Aplica as opcoes; devolve a base de analise e a cadeia de filtros."""
    d = base(o.unidade)
    cadeia = [("credito pessoal, 18 anos ou mais (amostra completa)", len(d), d[CLI].nunique())]
    if o.amostra == "janela":
        d = d[d["NA_JANELA"]]
        cadeia.append(("iniciados na janela jul/2019-jun/2020", len(d), d[CLI].nunique()))
    if o.max_parcelas:
        d = d[d["QTDE_TOTAL_PARCELAS"] <= o.max_parcelas]
        cadeia.append((f"ate {o.max_parcelas} parcelas", len(d), d[CLI].nunique()))
    if o.produto:
        d = d[d["GRUPO_PRODUTO_CONTRATADO"] == o.produto]
        cadeia.append((f"so o produto {o.produto}", len(d), d[CLI].nunique()))
    d = d[d["TAXA_ANUAL"].notna() & d["Score"].notna() & d["VLR_CONTRATADO"].notna()]
    cadeia.append(("com taxa, score e valor", len(d), d[CLI].nunique()))
    d = d.copy()
    renda = d["RENDA_CADASTRAL"].where(d["RENDA_CADASTRAL"] > 0)
    if o.sem_renda == "excluir":
        d = d[renda.notna()].copy()
        renda = renda[renda.notna()]
        cadeia.append(("excluidos contratos sem renda", len(d), d[CLI].nunique()))
    else:
        renda = renda.fillna(0.0)
        cadeia.append(("contratos sem renda mantidos com renda = 0", len(d), d[CLI].nunique()))
    d["RENDA"] = renda

    # limite de renda
    if o.limite_renda == "2SM":
        d["LIMITE"] = 2 * salario_minimo(d["MES_INICIO"])
    elif o.limite_renda == "2SM_cliente":  # limite do 1o contrato do cliente na amostra: o cliente nao troca de grupo
        lim = 2 * salario_minimo(d["MES_INICIO"])
        primeiro = d["DATA_INICIO_CONTRATO"].groupby(d[CLI]).transform("min")
        d["LIMITE"] = (2 * salario_minimo(primeiro.dt.to_period("M"))).values
    else:
        d["LIMITE"] = float(o.limite_renda)

    # grupos
    if o.atributo == "renda":
        g1 = (d["RENDA"] <= d["LIMITE"]) if o.limite_inclusivo else (d["RENDA"] < d["LIMITE"])
    else:
        a = ATR[o.atributo]
        x = d[a["coluna"]]
        ok = x.notna()
        if "validos" in a:
            ok &= x.isin(a["validos"])
        d = d[ok].copy()
        g1 = _regra(d[a["coluna"]], a["regra_G1"])
        cadeia.append((f"com {a['rotulo'].lower()} informado(a)", len(d), d[CLI].nunique()))
    d["G1"] = g1.astype(int)

    # tamanho do emprestimo
    if o.medida == "renda":
        with np.errstate(divide="ignore"):
            d["TAMANHO"] = np.where(d["RENDA"] > 0, (d["VLR_CONTRATADO"] / d["RENDA"]).round(3), np.inf)
        ed = FX["emprestimo_renda"]
    else:
        d["TAMANHO"] = d["VLR_CONTRATADO"]
        ed = faixas_valor(o.unidade)
    d["f_tam"] = pd.cut(d["TAMANHO"], ed, include_lowest=True).cat.codes
    fx_cel = FX["score"] if o.faixas_score == "antigas" else FX["score_banco"]
    fx_tab = FX["score_tabela"] if o.faixas_score == "antigas" else FX["score_banco"]
    d["f_score"] = pd.cut(d["Score"], fx_cel, include_lowest=True).cat.codes
    d["f_score_tab"] = pd.cut(d["Score"], fx_tab, include_lowest=True)
    d["f_score_tab"] = d["f_score_tab"].cat.rename_categories(lambda c: str(c).replace("(-0.001,", "[0,"))
    chave = ["f_score"] if o.medida == "nenhuma" else ["f_score", "f_tam"]
    if "parcelas" in o.controles:
        d["f_parc"] = pd.cut(d["QTDE_TOTAL_PARCELAS"].fillna(-1), [-np.inf] + FX["parcelas"][1:]).cat.codes
        chave.append("f_parc")
    if "default" in o.controles:
        d["f_def"] = (d["MAU_24"] > 0).astype(int)
        chave.append("f_def")
    if "faixa_renda" in o.controles and o.atributo != "renda":
        d["f_renda"] = ((d["RENDA"] <= d["LIMITE"]) if o.limite_inclusivo else (d["RENDA"] < d["LIMITE"])).astype(int)
        chave.append("f_renda")
    if "produto" in o.controles:
        d["f_prod"] = d["PRODUTO"].astype("category").cat.codes
        chave.append("f_prod")
    d["CELULA"] = d.groupby(chave, sort=True).ngroup()
    d["CLI_ID"] = d[CLI].astype("category").cat.codes
    d["BOM_SCORE"] = d["Score"] <= o.corte_score
    return d, cadeia


# ------------------------------------------------------------------ metricas

def _feols(d: pd.DataFrame, fe: bool) -> dict:
    import pyfixest as pf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fml = "TAXA_ANUAL ~ G1" + (" | CELULA" if fe else "")
        m = pf.feols(fml, data=d[["TAXA_ANUAL", "G1", "CELULA", "CLI_ID"]],
                     vcov={"CRV1": "CLI_ID"})
        t = m.tidy().loc["G1"]
    return {"est": float(t["Estimate"]), "ep": float(t["Std. Error"]),
            "ic_inf": float(t["2.5%"]), "ic_sup": float(t["97.5%"]),
            "p": float(t["Pr(>|t|)"]), "n": int(m._N)}


def _celulas(s1, n1, s2, n2, min_n):
    ok = (n1 >= min_n) & (n2 >= min_n) & (n1 > 0) & (n2 > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = s1 / n1 - s2 / n2
    w = (n1 + n2) * ok
    eq1 = gap[ok].mean() if ok.any() else np.nan
    eq2 = (gap[ok] * w[ok]).sum() / w[ok].sum() if ok.any() else np.nan
    return eq1, eq2, ok, gap, w


def fairness_celulas(d: pd.DataFrame, min_n: int, B: int, seed: int,
                     grupos_extra: pd.Series | None = None) -> dict:
    """Eqs. 1 e 2 com IC por bootstrap de clientes. `grupos_extra` (rotulo por
    contrato) produz tambem a Eq. 2 dentro de cada rotulo (ex.: faixa de score)."""
    nc = int(d["CELULA"].max()) + 1
    # int64: cat.codes pode vir como int8 e estourar (codigo * 2 > 127) em subamostras pequenas
    idx = d["CELULA"].to_numpy().astype(np.int64) * 2 + d["G1"].to_numpy().astype(np.int64)
    y = d["TAXA_ANUAL"].to_numpy()
    cli = d["CLI_ID"].to_numpy()
    s = np.bincount(idx, weights=y, minlength=2 * nc).reshape(nc, 2)
    n = np.bincount(idx, minlength=2 * nc).reshape(nc, 2).astype(float)
    eq1, eq2, ok, gap, w = _celulas(s[:, 1], n[:, 1], s[:, 0], n[:, 0], min_n)
    out = {"eq1": eq1, "eq2": eq2, "celulas_total": int(((n[:, 0] + n[:, 1]) > 0).sum()),
           "celulas_validas": int(ok.sum()),
           "contratos_em_celulas_validas": int(w.sum()),
           "contratos_G1_validos": int((n[:, 1] * ok).sum()),
           "contratos_G2_validos": int((n[:, 0] * ok).sum())}

    # rotulo da celula para Eq. 2 por grupo extra (cada celula pertence a um rotulo)
    lab_cel = None
    if grupos_extra is not None:
        codes = grupos_extra.cat.codes.to_numpy() if hasattr(grupos_extra, "cat") else grupos_extra.to_numpy()
        lab_cel = np.full(nc, -1)
        lab_cel[d["CELULA"].to_numpy().astype(np.int64)] = codes
        nl = len(grupos_extra.cat.categories) if hasattr(grupos_extra, "cat") else int(codes.max()) + 1

        def por_rotulo(gap_, w_, ok_):
            num = np.bincount(lab_cel[ok_], weights=(gap_ * w_)[ok_], minlength=nl)
            den = np.bincount(lab_cel[ok_], weights=w_[ok_], minlength=nl)
            with np.errstate(invalid="ignore", divide="ignore"):
                return num / den
        out["eq2_por_rotulo"] = por_rotulo(gap, w, ok)

    rng = np.random.default_rng(seed)
    ncli = int(cli.max()) + 1
    b1, b2, bl = [], [], []
    for _ in range(B):
        wc = rng.poisson(1.0, ncli)[cli].astype(float)
        sb = np.bincount(idx, weights=y * wc, minlength=2 * nc).reshape(nc, 2)
        nb = np.bincount(idx, weights=wc, minlength=2 * nc).reshape(nc, 2)
        e1, e2, okb, gb, wb = _celulas(sb[:, 1], nb[:, 1], sb[:, 0], nb[:, 0], min_n)
        b1.append(e1)
        b2.append(e2)
        if lab_cel is not None:
            bl.append(por_rotulo(gb, wb, okb))
    if B:
        out["eq1_ic"] = tuple(np.nanpercentile(b1, [2.5, 97.5]))
        out["eq2_ic"] = tuple(np.nanpercentile(b2, [2.5, 97.5]))
        if lab_cel is not None:
            bl = np.array(bl)
            out["eq2_por_rotulo_ic"] = np.nanpercentile(bl, [2.5, 97.5], axis=0)
    return out


def wdi(d: pd.DataFrame) -> dict:
    ed = FX["taxa_wdi"]
    f = pd.cut(d["TAXA_ANUAL"], ed, include_lowest=True)
    c = pd.crosstab(f, d["G1"]).reindex(columns=[0, 1], fill_value=0)
    dom, disc = c[0].to_numpy(float), c[1].to_numpy(float)
    i = np.arange(1, len(c) + 1)
    w = np.exp(-i.astype(float))
    ok = (dom + disc) > 0
    r = (dom - disc) / np.where(ok, dom + disc, 1)
    pd_, pc = dom / dom.sum(), disc / disc.sum()
    rs = (pd_ - pc) / np.where((pd_ + pc) > 0, pd_ + pc, 1)
    k = dom.sum() / disc.sum() if disc.sum() else np.nan
    return {"wdi": float((r * w)[ok].sum() / w[ok].sum()),
            "wdi_referencia_sem_disparidade": float((k - 1) / (k + 1)),
            "wdi_proporcoes": float((rs * w)[ok].sum() / w[ok].sum()),
            "wdi_faixas": [str(x) for x in c.index],
            "wdi_contagens_G2": dom.astype(int).tolist(),
            "wdi_contagens_G1": disc.astype(int).tolist()}


def por_faixa_score(d: pd.DataFrame, fc: dict) -> pd.DataFrame:
    g = d.groupby(["f_score_tab", "G1"], observed=False)["TAXA_ANUAL"]
    t = g.agg(["size", "mean", "std"]).unstack("G1")
    t.columns = [f"{a}_{'G1' if b else 'G2'}" for a, b in t.columns]
    t = t.rename(columns={"size_G1": "n_G1", "size_G2": "n_G2", "mean_G1": "taxa_G1",
                          "mean_G2": "taxa_G2", "std_G1": "dp_G1", "std_G2": "dp_G2"})
    for dfn in ("MAU_24",):
        m = d.groupby(["f_score_tab", "G1"], observed=False)[dfn].apply(lambda s: (s > 0).mean()).unstack("G1")
        t[f"{dfn}_G1"], t[f"{dfn}_G2"] = m[1], m[0]
    t["dif_bruta"] = t["taxa_G1"] - t["taxa_G2"]
    t["dif_celulas_eq2"] = fc.get("eq2_por_rotulo", np.full(len(t), np.nan))[: len(t)]
    if "eq2_por_rotulo_ic" in fc:
        t["dif_celulas_ic_inf"] = fc["eq2_por_rotulo_ic"][0][: len(t)]
        t["dif_celulas_ic_sup"] = fc["eq2_por_rotulo_ic"][1][: len(t)]
    t.index = [str(x) for x in t.index]
    t.index.name = "faixa_score"
    return t.reset_index()


def teste_por_faixa(d: pd.DataFrame) -> pd.DataFrame:
    """Teste t da diferenca G1 - G2 dentro de cada faixa de score: regressao com efeitos fixos
    de celula na faixa, erros agrupados por cliente; p-valores corrigidos por Holm entre faixas."""
    rows = []
    for fx, s in d.groupby("f_score_tab", observed=True):
        if s["G1"].nunique() < 2:
            continue
        mix = s.groupby("CELULA")["G1"].agg(["min", "max"])
        if not ((mix["min"] == 0) & (mix["max"] == 1)).any():
            continue  # nenhuma celula com os dois grupos
        s = s.copy()
        s["CELULA"] = s["CELULA"].astype("category").cat.codes
        try:
            x = _feols(s, fe=True)
        except KeyboardInterrupt:
            raise
        except BaseException:  # pyfixest pode falhar (panic) em faixas muito pequenas
            continue
        rows.append({"faixa_score": str(fx), "dif_reg": x["est"], "dif_reg_ic_inf": x["ic_inf"],
                     "dif_reg_ic_sup": x["ic_sup"], "t": x["est"] / x["ep"] if x["ep"] else np.nan, "p": x["p"]})
    t = pd.DataFrame(rows)
    if len(t):
        m = len(t)
        ordem = np.argsort(t["p"].to_numpy())
        adj = np.empty(m)
        run = 0.0
        for k, i in enumerate(ordem):
            run = max(run, min(1.0, (m - k) * t["p"].iloc[i]))
            adj[i] = run
        t["p_holm"] = adj
    return t


def analisar(o: Opcoes) -> dict:
    d, cadeia = preparar(o)
    r = {"opcoes": o.descricao(), "tag": o.tag(),
         "cadeia": pd.DataFrame(cadeia, columns=["etapa", "contratos", "clientes"])}
    a = ATR[o.atributo]
    r["rotulos"] = {"G1": a["G1"], "G2": a["G2"], "G1_en": a["G1_en"], "G2_en": a["G2_en"],
                    "atributo": a["rotulo"]}
    cont = {"contratos": len(d), "clientes": int(d[CLI].nunique()),
            "contratos_G1": int(d["G1"].sum()), "contratos_G2": int((1 - d["G1"]).sum()),
            "clientes_G1": int(d.loc[d["G1"] == 1, CLI].nunique()),
            "clientes_G2": int(d.loc[d["G1"] == 0, CLI].nunique()),
            "taxa_media_G1": float(d.loc[d["G1"] == 1, "TAXA_ANUAL"].mean()),
            "taxa_media_G2": float(d.loc[d["G1"] == 0, "TAXA_ANUAL"].mean())}
    if o.atributo == "renda":
        cont["limite_renda_min"] = float(d["LIMITE"].min())
        cont["limite_renda_max"] = float(d["LIMITE"].max())
        cont["contratos_renda_zero"] = int((d["RENDA"] == 0).sum())
    r["contagens"] = cont
    res = {}
    for nome, sub in (("todos", d), ("bom_score", d[d["BOM_SCORE"]])):
        if sub["G1"].nunique() < 2:
            continue
        sub = sub.copy()
        sub["CELULA"] = sub["CELULA"].astype("category").cat.codes
        sub["CLI_ID"] = sub["CLI_ID"].astype("category").cat.codes
        fc = fairness_celulas(sub, o.min_n, o.B, o.seed,
                              sub["f_score_tab"] if nome == "todos" else None)
        res[nome] = {"contratos": len(sub), "clientes": int(sub[CLI].nunique()),
                     "contratos_G1": int(sub["G1"].sum()), "contratos_G2": int((1 - sub["G1"]).sum()),
                     "taxa_media_G1": float(sub.loc[sub["G1"] == 1, "TAXA_ANUAL"].mean()),
                     "taxa_media_G2": float(sub.loc[sub["G1"] == 0, "TAXA_ANUAL"].mean()),
                     "dif_bruta": _feols(sub, fe=False), "dif_celulas_reg": _feols(sub, fe=True),
                     **{k: v for k, v in fc.items() if not k.startswith("eq2_por_rotulo")},
                     **wdi(sub)}
        if nome == "todos":
            r["por_score"] = por_faixa_score(sub, fc).merge(teste_por_faixa(sub), on="faixa_score", how="left")
    r["resultados"] = res
    return r


def resumo_linha(r: dict, grupo: str = "todos") -> dict:
    """Uma linha plana (para CSV e tabelas comparativas)."""
    x = r["resultados"].get(grupo, {})
    o = r["opcoes"]
    row = {"tag": r["tag"], "amostra": o["amostra"], "sem_renda": o["sem_renda"],
           "medida": o["medida"], "controles": o["controles"], "atributo": o["atributo"],
           "limite_renda": o["limite_renda"], "corte_score": o["corte_score"],
           "subamostra": grupo}
    if not x:
        return row
    row.update({"contratos": x["contratos"], "clientes": x["clientes"],
                "contratos_G1": x["contratos_G1"], "contratos_G2": x["contratos_G2"],
                "taxa_G1": x["taxa_media_G1"], "taxa_G2": x["taxa_media_G2"]})
    for k in ("dif_bruta", "dif_celulas_reg"):
        row[k] = x[k]["est"]
        row[f"{k}_ic_inf"] = x[k]["ic_inf"]
        row[f"{k}_ic_sup"] = x[k]["ic_sup"]
        row[f"{k}_p"] = x[k]["p"]
    for k in ("eq1", "eq2"):
        row[k] = x[k]
        if f"{k}_ic" in x:
            row[f"{k}_ic_inf"], row[f"{k}_ic_sup"] = x[f"{k}_ic"]
    row.update({"celulas_validas": x["celulas_validas"],
                "contratos_em_celulas_validas": x["contratos_em_celulas_validas"],
                "wdi": x["wdi"], "wdi_referencia": x["wdi_referencia_sem_disparidade"],
                "wdi_proporcoes": x["wdi_proporcoes"]})
    return row
