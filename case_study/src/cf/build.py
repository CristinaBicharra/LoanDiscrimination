"""Monta a base por contrato a partir do painel bruto do banco.

Decisoes dos autores (23/11/2026) aplicadas aqui:
  1. O painel tem uma linha por contrato x mes (transacoes). Agrupa-se para
     obter um registro por contrato e cliente.
  2. Contrato = COD_CONTRATO_MASCARADO + VLR_CONTRATADO (opcao 'id_valor', padrao)
     ou so COD_CONTRATO_MASCARADO (opcao 'id', como no codigo de 2023): quando o mesmo ID
     aparece com valores iniciais distintos, sao contratos distintos.
     Os atributos do contrato vem da 1a observacao mensal (menor
     DATA_REFERENCIA) com aquele valor.
  3. So credito pessoal (regra de classe de produto do notebook de 2023), tudo
     agrupado; consignado, cheque especial, agro e outros ficam de fora.
  4. Excluem-se tomadores com menos de 18 anos.
  5. Marca-se a janela: DATA_INICIO_CONTRATO entre 01/07/2019 e 30/06/2020.

Inadimplencia: MAU_24, indicador do cliente em 24 meses (ultima observacao do
contrato). Dias de atraso NAO sao usados: segundo o banco, o atraso e mal
reportado (contratos em atraso eram repactuados ou passavam a ter juros de
mora, e esses eventos nao estao nos dados).

Saidas (CONFIDENCIAIS, ficam em data/derived/, fora de qualquer repositorio):
  contratos_cp.parquet  - uma linha por contrato de credito pessoal, 18+
  build_stats.json      - contagens agregadas da cadeia amostral

    python3 -m src.cf.build      (a partir da pasta do projeto; ~5 min)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
def settings() -> dict:
    """Configuracao especifica do banco (arquivo, regras de produto, localizacao), fora do codigo:
    config/private_settings.json (local, nao publicado) ou, na falta dele, o exemplo publico."""
    for nome in ("private_settings.json", "private_settings.example.json"):
        p = ROOT / "config" / nome
        if p.exists():
            return json.loads(p.read_text())
    return {}


RAW = ROOT / "data" / settings().get("raw_file", "bank_panel.csv")
DERIVED = ROOT / "data" / "derived"
UNIDADES = {"id_valor": "ID + valor contratado (decisao 4)", "id": "so ID (codigo original de 2023)"}


def caminho(unidade: str = "id_valor") -> Path:
    return DERIVED / ("contratos_cp.parquet" if unidade == "id_valor" else f"contratos_cp_{unidade}.parquet")


STATS = DERIVED / "build_stats.json"

ID, CLI, VAL = "COD_CONTRATO_MASCARADO", "CHAVE_ALEATORIA", "VLR_CONTRATADO"
JANELA = (pd.Timestamp("2019-07-01"), pd.Timestamp("2020-07-01"))  # [ini, fim)

STR = [ID, CLI, "GRUPO_PRODUTO_CONTRATADO", "DESCR_PRODUTO_GESTAO",
       "CEP_RESIDENCIAL", "UF_RESIDENCIAL"]
DATES = ["DATA_REFERENCIA", "DATA_INICIO_CONTRATO",
         "DATA_REFERENCIA_MES_ORIGINACAO_SAFRA"]
NUM = [VAL, "SALDO_CONTABIL", "QTDE_TOTAL_PARCELAS", "NUM_PARCELA_PAGA",
       "TAXA_MENSAL_OPERACAO", "PERIODICIDADE_DIAS", "MAU_24",
       "IDADE", "COD_ESTADO_CIVIL", "COD_ESCOLARIDADE", "SEXO", "RENDA_CADASTRAL",
       "IND_FOLHA_BCO", "FUNC_PUB_ATIVIDADE", "Score"]

def classe_produto(g) -> str:
    """Regra do notebook de 2023 (reproduz a Tabela 1 do artigo): a primeira regra cujo trecho aparece
    no grupo de produto define a classe. As regras ficam em config/private_settings.json."""
    x = "" if g is None or pd.isna(g) else str(g)
    for trecho, classe in settings().get("product_rules", []):
        if trecho in x:
            return classe
    return "Outros"


def _num(s: pd.Series) -> pd.Series:
    s = s.str.strip().replace({"": None, "NA": None})
    return pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")


def ler_painel() -> pd.DataFrame:
    parts = []
    for ch in pd.read_csv(RAW, sep=";", usecols=STR + DATES + NUM, dtype=str,
                          chunksize=500_000, encoding="utf-8-sig",
                          keep_default_na=False):
        o = pd.DataFrame(index=ch.index)
        for c in STR:
            o[c] = ch[c].str.strip()
        for c in DATES:
            o[c] = pd.to_datetime(ch[c].str.strip(), format="%m/%d/%Y %H:%M:%S",
                                  errors="coerce")
        for c in NUM:
            o[c] = _num(ch[c])
        parts.append(o)
    p = pd.concat(parts, ignore_index=True)
    for c in STR:
        p[c] = p[c].astype("category")
    return p


def montar(p: pd.DataFrame, unidade: str = "id_valor") -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    p = p.copy()
    st = {"painel_linhas": len(p), "ids_distintos": int(p[ID].nunique()),
          "clientes_distintos": int(p[CLI].nunique())}
    cls = {c: classe_produto(c) for c in p["GRUPO_PRODUTO_CONTRATADO"].cat.categories}
    p = p.sort_values([ID, "DATA_REFERENCIA"], kind="mergesort").reset_index(drop=True)

    # unidade = (ID, valor contratado); NaN de valor vira sua propria unidade
    p["_val"] = p[VAL].fillna(-1.0) if unidade == "id_valor" else 0.0
    g = p.groupby([ID, "_val"], observed=True, sort=False)
    p["_unidade"] = g.ngroup()

    first = p.groupby("_unidade", sort=False).head(1).set_index("_unidade")
    gu = p.groupby("_unidade", sort=False)
    first["MAU_24"] = gu["MAU_24"].last()
    first["N_MESES"] = gu.size()
    first["MAX_PARCELA_PAGA"] = gu["NUM_PARCELA_PAGA"].max()
    first["N_VALORES_NO_ID"] = first.groupby(ID, observed=True)[VAL].transform("size")
    first["CLASSE"] = first["GRUPO_PRODUTO_CONTRATADO"].astype(str).map(cls)
    first = first.drop(columns=["_val", "DATA_REFERENCIA"]).reset_index(drop=True)

    st["unidade"] = UNIDADES[unidade]
    st["contratos_todos_produtos"] = len(first)
    st["contratos_por_classe"] = first["CLASSE"].value_counts().to_dict()
    cp = first[first["CLASSE"] == "Credito_Pessoal"].copy()
    st["cp"] = len(cp)
    st["cp_menores_18"] = int((cp["IDADE"] < 18).sum())
    st["cp_inadimplentes_MAU24_todas_idades"] = int((cp["MAU_24"] > 0).sum())
    st["cp_idade_ausente"] = int(cp["IDADE"].isna().sum())
    cp = cp[cp["IDADE"] >= 18].copy()
    st["cp_18"] = len(cp)

    cp["NA_JANELA"] = (cp["DATA_INICIO_CONTRATO"] >= JANELA[0]) & (cp["DATA_INICIO_CONTRATO"] < JANELA[1])
    cp["TAXA_ANUAL"] = ((1 + cp["TAXA_MENSAL_OPERACAO"] / 100) ** 12 - 1) * 100
    cp["SEM_RENDA"] = ~(cp["RENDA_CADASTRAL"] > 0)
    for nome, sub in (("completa", cp), ("janela", cp[cp["NA_JANELA"]])):
        st[f"{nome}_contratos"] = len(sub)
        st[f"{nome}_clientes"] = int(sub[CLI].nunique())
        st[f"{nome}_sem_renda"] = int(sub["SEM_RENDA"].sum())
        st[f"{nome}_com_renda"] = int((~sub["SEM_RENDA"]).sum())
        st[f"{nome}_clientes_com_renda"] = int(sub.loc[~sub["SEM_RENDA"], CLI].nunique())
        st[f"{nome}_taxa_ausente"] = int(sub["TAXA_ANUAL"].isna().sum())
        st[f"{nome}_score_ausente"] = int(sub["Score"].isna().sum())
    # MAU_24 deveria ser constante por cliente
    v = cp.groupby(CLI, observed=True)["MAU_24"].nunique()
    st["clientes_com_MAU_24_variando"] = int((v > 1).sum())
    st["segundos"] = round(time.time() - t0)

    for c in (ID, CLI, "GRUPO_PRODUTO_CONTRATADO", "DESCR_PRODUTO_GESTAO",
              "CEP_RESIDENCIAL", "UF_RESIDENCIAL"):
        cp[c] = cp[c].astype(str)
    return cp.reset_index(drop=True), st


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    p = ler_painel()
    todos = {}
    for u in UNIDADES:
        cp, st = montar(p, u)
        cp.to_parquet(caminho(u), index=False)
        todos[u] = st
    STATS.write_text(json.dumps(todos, indent=2, ensure_ascii=False, default=int))
    print(json.dumps(todos, indent=2, ensure_ascii=False, default=int))


if __name__ == "__main__":
    main()
