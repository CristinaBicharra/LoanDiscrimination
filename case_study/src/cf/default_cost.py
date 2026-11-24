"""Etapa 3: inadimplencia por grupo e custo da equidade.

Inadimplencia = MAU_24, indicador do CLIENTE em 24 meses ("defaulted on that or
on another loan"); no nivel do contrato, repete o indicador do cliente. Dias de
atraso nao sao usados (mal reportados pelo banco: repactuacao e juros de mora
nao estao nos dados).

(a) Inadimplencia de G1 e G2 (no atributo escolhido; o artigo usa renda), em
    tres subamostras: todos, score <= corte e score > corte. Para cada uma:
    no agregado e na mesma faixa de score (efeitos fixos de faixa), no nivel do
    contrato (IC com erros agrupados por cliente) e no nivel do cliente (um
    registro por cliente: grupo e score do primeiro contrato da subamostra;
    inadimplente se o indicador do cliente for 1). Tambem por faixa de score,
    com o numero e a participacao dos contratos de cada grupo em cada faixa.

(b) Custo de reduzir alpha = 10%, ..., 100% da diferenca de taxas.
    * Diferenca de cada celula (score x tamanho [x controles]) = taxa media G1 -
      taxa media G2. A politica reduz a taxa anual dos contratos de G1 em
      alpha x diferenca, so nas celulas em que G1 paga mais (diferenca > 0) e
      que tem os dois grupos; nenhuma taxa sobe; taxa minima 0.
    * Juros de cada contrato pela tabela Price (taxa mensal equivalente a taxa
      anual, numero de parcelas do contrato, parcelas mensais):
      juros = n x PMT - valor.
    * Receita de juros esperada = (1 - PD) x juros. Custo = queda da receita
      esperada. Custo em % da receita de juros esperada da propria faixa de
      score (uma curva por faixa) e, no total, das faixas com score <= corte
      em % da receita de toda a carteira.

(c) Duas versoes de PD:
    * "score": PD = score do contrato (default implicito no score);
    * "observado": PD = taxa de inadimplencia observada (MAU_24) do grupo
      (G1 ou G2) na faixa de score.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .analysis import CLI, Opcoes, preparar

ALPHAS = np.round(np.arange(0, 1.0001, 0.1), 2)
Y = "Y_MAU_24"


def _feols(d, y, fe=None):
    import pyfixest as pf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pf.feols(f"{y} ~ G1" + (f" | {fe}" if fe else ""), data=d, vcov={"CRV1": "CLI_ID"})
    t = m.tidy().loc["G1"]
    _feols.ultimo_p = float(t["Pr(>|t|)"])
    return float(t["Estimate"]), float(t["2.5%"]), float(t["97.5%"])


def _holm(p):
    p = np.asarray(p, float)
    ordem = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for k, i in enumerate(ordem):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def _bloco(sub: pd.DataFrame, nome: str, nivel: str) -> dict:
    sub = sub.copy()
    sub["CLI_ID"] = sub["CLI_KEY"].astype("category").cat.codes
    e, lo, hi = _feols(sub[[Y, "G1", "CLI_ID"]], Y)
    out = {"subamostra": nome, "nivel": nivel, "n": len(sub), "n_G1": int(sub.G1.sum()),
           "n_G2": int((1 - sub.G1).sum()), "G1": sub.loc[sub.G1 == 1, Y].mean(),
           "G2": sub.loc[sub.G1 == 0, Y].mean(), "dif": e, "dif_ic_inf": lo, "dif_ic_sup": hi}
    if sub["FS"].nunique() > 1:
        ef, lof, hif = _feols(sub[[Y, "G1", "CLI_ID", "FS"]], Y, "FS")
    else:
        ef, lof, hif = e, lo, hi
    out["p_mesma_faixa_score"] = _feols.ultimo_p
    out.update({"dif_mesma_faixa_score": ef, "dif_mesma_faixa_score_ic_inf": lof,
                "dif_mesma_faixa_score_ic_sup": hif})
    return out


# ------------------------------------------------------------------ (a)

def inadimplencia(o: Opcoes) -> dict:
    d, _ = preparar(o)
    d = d.copy()
    d[Y] = (d["MAU_24"] > 0).astype(float) * 100
    d["FS"] = d["f_score_tab"].cat.codes
    d["CLI_KEY"] = d[CLI]
    subs = {"todos": d, f"score <= {o.corte_score}": d[d.BOM_SCORE],
            f"score > {o.corte_score}": d[~d.BOM_SCORE]}
    ag, faixas = [], []
    for nome, s in subs.items():
        if s.G1.nunique() < 2:
            continue
        ag.append(_bloco(s, nome, "contrato"))
        c = s.sort_values("DATA_INICIO_CONTRATO").groupby(CLI, observed=True).agg(
            G1=("G1", "first"), FS=("FS", "first"), Y_MAU_24=(Y, "max")).reset_index()
        c["CLI_KEY"] = c[CLI]
        ag.append(_bloco(c, nome, "cliente"))
    nG1, nG2 = d.G1.sum(), (1 - d.G1).sum()
    for nivel in ("contrato", "cliente"):
        base = d if nivel == "contrato" else d.sort_values("DATA_INICIO_CONTRATO").groupby(
            CLI, observed=True).agg(G1=("G1", "first"), f_score_tab=("f_score_tab", "first"),
                                    Y_MAU_24=(Y, "max")).reset_index()
        base = base.copy()
        base["CLI_ID"] = base[CLI].astype("category").cat.codes
        tG1, tG2 = base.G1.sum(), (1 - base.G1).sum()
        for fx, sub in base.groupby("f_score_tab", observed=True):
            if sub.G1.nunique() < 2:
                continue
            e, lo, hi = _feols(sub[[Y, "G1", "CLI_ID"]], Y)
            faixas.append({"nivel": nivel, "faixa_score": str(fx),
                           "n_G1": int(sub.G1.sum()), "n_G2": int((1 - sub.G1).sum()),
                           "pct_dos_G1": 100 * sub.G1.sum() / tG1, "pct_dos_G2": 100 * (1 - sub.G1).sum() / tG2,
                           "pct_do_total": 100 * len(sub) / len(base),
                           "G1": sub.loc[sub.G1 == 1, Y].mean(), "G2": sub.loc[sub.G1 == 0, Y].mean(),
                           "dif": e, "dif_ic_inf": lo, "dif_ic_sup": hi, "p": _feols.ultimo_p})
    calib = d.groupby("f_score_tab", observed=True).agg(
        score_medio=("Score", "mean"), MAU_24=(Y, "mean"), n=("Score", "size")).reset_index().rename(
        columns={"f_score_tab": "faixa_score"})
    calib["score_medio"] *= 100
    calib["faixa_score"] = calib["faixa_score"].astype(str)
    pf_ = pd.DataFrame(faixas)
    pf_["p_holm"] = np.nan
    for nivel in ("contrato", "cliente"):
        m = pf_.nivel == nivel
        pf_.loc[m, "p_holm"] = _holm(pf_.loc[m, "p"])
        # Holm so entre as faixas com score <= corte (as que importam para a politica)
        rr = pf_.loc[m, "faixa_score"].map(lambda x: float(str(x).split(",")[1].strip(" ]")))
        mb = m & (pf_["faixa_score"].map(lambda x: float(str(x).split(",")[1].strip(" ]"))) <= o.corte_score + 1e-9)
        pf_.loc[mb, "p_holm_ate_corte"] = _holm(pf_.loc[mb, "p"])
    faixas = pf_
    acima = ~d.BOM_SCORE
    part = {"pct_contratos_acima_corte": 100 * acima.mean(),
            "pct_G1_acima_corte": 100 * (acima & (d.G1 == 1)).sum() / nG1,
            "pct_G2_acima_corte": 100 * (acima & (d.G1 == 0)).sum() / nG2,
            "contratos_acima_corte": int(acima.sum())}
    return {"agregado": pd.DataFrame(ag), "por_faixa": faixas,
            "calibracao": calib, "participacao": part, "corte": o.corte_score}


# ------------------------------------------------------------------ (b, c)

def juros_price(valor, taxa_anual, n):
    m = (1 + np.clip(taxa_anual, 0, None) / 100) ** (1 / 12) - 1
    n = np.clip(np.nan_to_num(n, nan=1), 1, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmt = np.where(m > 0, valor * m / (1 - (1 + m) ** (-n)), valor / n)
    return n * pmt - valor


def custo(o: Opcoes) -> dict:
    d, _ = preparar(o)
    d = d.copy()
    g = d.groupby(["CELULA", "G1"])["TAXA_ANUAL"].agg(["mean", "size"]).unstack("G1")
    ok = (g[("size", 1)] >= o.min_n) & (g[("size", 0)] >= o.min_n)
    gap = (g[("mean", 1)] - g[("mean", 0)]).where(ok)
    d["GAP_CEL"] = d["CELULA"].map(gap)
    alvo = (d["G1"] == 1) & (d["GAP_CEL"] > 0)
    d["FAIXA"] = d["f_score_tab"].astype(str)
    faixas = list(d["f_score_tab"].cat.categories.astype(str))
    rights = [float(s.split(",")[1].strip(" ]")) for s in faixas]
    d["BOM"] = d["Score"] <= o.corte_score

    obs = (d["MAU_24"] > 0).astype(float)
    versoes = {"score": d["Score"].clip(0, 1),
               "observado": obs.groupby([d["FAIXA"], d["G1"]]).transform("mean")}

    I0 = juros_price(d["VLR_CONTRATADO"].to_numpy(), d["TAXA_ANUAL"].to_numpy(),
                     d["QTDE_TOTAL_PARCELAS"].to_numpy())
    curvas, resumo = [], []
    for v, PD in versoes.items():
        surv = 1 - PD.to_numpy()
        R0 = surv * I0
        rec_faixa = pd.Series(R0).groupby(d["FAIXA"].to_numpy()).sum()
        rec_total = R0.sum()
        bom = d["BOM"].to_numpy()
        for a in ALPHAS:
            r1 = d["TAXA_ANUAL"].to_numpy() - np.where(alvo, a * d["GAP_CEL"].fillna(0).to_numpy(), 0)
            I1 = juros_price(d["VLR_CONTRATADO"].to_numpy(), r1, d["QTDE_TOTAL_PARCELAS"].to_numpy())
            dc = surv * (I0 - I1)
            cf = pd.Series(dc).groupby(d["FAIXA"].to_numpy()).sum()
            for fx in faixas:
                curvas.append({"versao_pd": v, "alpha": a, "faixa_score": fx,
                               "custo_R$": cf.get(fx, 0.0), "receita_faixa_R$": rec_faixa.get(fx, np.nan),
                               "custo_pct_receita_faixa": 100 * cf.get(fx, 0.0) / rec_faixa.get(fx, np.nan)})
            curvas.append({"versao_pd": v, "alpha": a, "faixa_score": f"score <= {o.corte_score}",
                           "custo_R$": dc[bom].sum(), "receita_faixa_R$": rec_total,
                           "custo_pct_receita_faixa": 100 * dc[bom].sum() / rec_total})
            curvas.append({"versao_pd": v, "alpha": a, "faixa_score": "todas",
                           "custo_R$": dc.sum(), "receita_faixa_R$": rec_total,
                           "custo_pct_receita_faixa": 100 * dc.sum() / rec_total})
        for fx in faixas:
            s = d["FAIXA"] == fx
            resumo.append({"versao_pd": v, "faixa_score": fx,
                           "contratos": int(s.sum()), "contratos_G1_com_reducao": int((s & alvo).sum()),
                           "reducao_media_pp": d.loc[s & alvo, "GAP_CEL"].mean(),
                           "PD_G1": 100 * PD[s & (d.G1 == 1)].mean(), "PD_G2": 100 * PD[s & (d.G1 == 0)].mean()})
    cv = pd.DataFrame(curvas)
    rs = pd.DataFrame(resumo)
    um = cv[cv.alpha == 1.0].pivot(index="faixa_score", columns="versao_pd",
                                   values=["custo_R$", "custo_pct_receita_faixa", "receita_faixa_R$"])
    um.columns = [f"{a}__{b}" for a, b in um.columns]
    rs = rs.merge(um.reset_index(), on="faixa_score", how="left")
    return {"curvas": cv, "resumo_alpha1": rs, "faixas": faixas, "rights": rights,
            "corte": o.corte_score,
            "receita_total": {v: float((1 - p.to_numpy()) @ I0) for v, p in versoes.items()},
            "contratos_com_reducao": int(alvo.sum())}
