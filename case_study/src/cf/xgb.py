"""Robustez sem faixas: contrafactual com XGBoost.

Pergunta: quanto o grupo G1 paga a mais do que pagaria se a regra de preco aplicada
ao grupo G2 fosse aplicada a ele?

1. Treina-se um XGBoost so com contratos de G2 para prever a taxa anual a partir
   do score e do tamanho do emprestimo (emprestimo/renda), as mesmas variaveis da
   celula C2, mas sem faixas: o modelo capta curvas e interacoes.
   Hiperparametros: grade pequena e fixa, escolhida por validacao cruzada em 5
   partes agrupadas por cliente (GroupKFold), so com G2.
2. Aplica-se o modelo aos contratos de G1. Diferenca = media de (taxa observada -
   taxa prevista) em G1. A mesma conta em G2 fora da amostra (previsoes de
   validacao cruzada) deve dar perto de zero.
3. IC 95%: bootstrap por cliente da diferenca media (modelo fixo).
4. Versao inversa (treina em G1, aplica em G2; sinal trocado) para simetria.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .analysis import Opcoes, preparar

GRADE = [dict(max_depth=d, learning_rate=0.05, n_estimators=n, min_child_weight=50,
              subsample=0.8, colsample_bytree=1.0)
         for d in (3, 5) for n in (200, 400)]


def _feats(d: pd.DataFrame) -> np.ndarray:
    tam = d["TAMANHO"].replace(np.inf, np.nan)
    tam = tam.fillna(tam.max() * 2 if tam.notna().any() else 0).clip(upper=tam.quantile(0.999))
    return np.column_stack([d["Score"].to_numpy(), tam.to_numpy()])


def _modelo(params, seed):
    from xgboost import XGBRegressor
    return XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=seed, n_jobs=4, **params)


def contrafactual(o: Opcoes, treino: int = 0, B: int = 200, seed: int = 20261123) -> dict:
    from sklearn.model_selection import GroupKFold
    d, _ = preparar(o)
    d = d.reset_index(drop=True)
    X, y = _feats(d), d["TAXA_ANUAL"].to_numpy()
    tr = (d["G1"] == treino).to_numpy()
    grupos = d["CLI_ID"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    melhor, melhor_rmse, oof_melhor = None, np.inf, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for params in GRADE:
            oof = np.empty(tr.sum())
            for a, b in gkf.split(X[tr], y[tr], grupos[tr]):
                m = _modelo(params, seed).fit(X[tr][a], y[tr][a])
                oof[b] = m.predict(X[tr][b])
            rmse = float(np.sqrt(np.mean((y[tr] - oof) ** 2)))
            if rmse < melhor_rmse:
                melhor, melhor_rmse, oof_melhor = params, rmse, oof
        m = _modelo(melhor, seed).fit(X[tr], y[tr])
    pred = m.predict(X[~tr])
    res_out = y[~tr] - pred                 # grupo contrafactual
    res_in = y[tr] - oof_melhor             # grupo de treino, fora da amostra
    sinal = 1 if treino == 0 else -1        # sempre G1 - G2
    r2 = 1 - np.mean(res_in ** 2) / np.var(y[tr])
    # bootstrap por cliente
    cli_out = pd.Series(grupos[~tr]).astype("category").cat.codes.to_numpy()
    rng = np.random.default_rng(seed)
    n_cli = cli_out.max() + 1
    soma = np.bincount(cli_out, weights=res_out, minlength=n_cli)
    cont = np.bincount(cli_out, minlength=n_cli).astype(float)
    bs = []
    for _ in range(B):
        w = rng.poisson(1.0, n_cli)
        bs.append((w @ soma) / (w @ cont))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    est = res_out.mean()
    if sinal < 0:
        est, lo, hi = -est, -hi, -lo
    bom = d.loc[~tr, "BOM_SCORE"].to_numpy()
    return {"treinado_em": "G2" if treino == 0 else "G1", "contratos_treino": int(tr.sum()),
            "contratos_aplicado": int((~tr).sum()), "params": melhor, "rmse_cv": melhor_rmse, "r2_cv": float(r2),
            "residuo_medio_treino_cv": float(res_in.mean()),
            "dif_G1_menos_G2": float(est), "ic_inf": float(lo), "ic_sup": float(hi),
            "dif_score_ate_corte": float(sinal * res_out[bom].mean()) if bom.any() else np.nan}


def tabela(amostras=("janela", "completa"), atributos=("renda",), B=200) -> pd.DataFrame:
    rows = []
    for am in amostras:
        for at in atributos:
            for treino in (0, 1):
                x = contrafactual(Opcoes(amostra=am, atributo=at), treino=treino, B=B)
                x["params"] = str(x["params"])
                rows.append({"amostra": am, "atributo": at, **x})
    return pd.DataFrame(rows)


def tabela_tex(t: pd.DataFrame) -> str:
    lin = []
    for r in t.itertuples():
        lin.append(f"{'Sampling window' if r.amostra == 'janela' else 'Full sample'} & "
                   f"{'Higher-income (G2)' if r.treinado_em == 'G2' else 'Lower-income (G1)'} & "
                   f"{r.contratos_treino:,} & {r.r2_cv:.2f} & {r.dif_G1_menos_G2:.1f} [{r.ic_inf:.1f}; {r.ic_sup:.1f}] & "
                   f"{r.dif_score_ate_corte:.1f} \\\\")
    return ("\\begin{table}[htbp]\n\\centering\\scriptsize\n"
            "\\caption{Robustness without bins: gradient-boosted trees (XGBoost) predict the annual interest rate from the "
            "credit score and the loan-to-income ratio, trained on one income group and applied to the other. Difference = "
            "mean of observed minus predicted rate, expressed as lower-income minus higher-income (percentage points); 95\\% "
            "CI from a client-level bootstrap. $R^2$ from 5-fold cross-validation grouped by client within the training group; "
            "hyperparameters chosen in the same cross-validation from a small fixed grid.}\n"
            "\\begin{tabular}{llrrcc}\n\\toprule\n"
            "Sample & Model trained on & Contracts & CV $R^2$ & Difference [95\\% CI] & Score $\\leq$ 0.4 \\\\\n\\midrule\n"
            + "\n".join(lin) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
