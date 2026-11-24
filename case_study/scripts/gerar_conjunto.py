"""Gera de uma vez o conjunto do artigo (analise principal + robustez, Etapas 2 a 4; ~6 min).

    python3 scripts/gerar_conjunto.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cf.export import gerar_conjunto  # noqa: E402
from src.cf.export3 import gerar_etapa3  # noqa: E402
from src.cf.etapa4 import gerar_etapa4  # noqa: E402
from src.cf import xgb  # noqa: E402
from src.cf import paper_tables  # noqa: E402

if __name__ == "__main__":
    log = lambda f, m: print(f"[{f:4.0%}] {m}", flush=True)  # noqa: E731
    p = gerar_conjunto(log)
    gerar_etapa3(p, log)
    gerar_etapa4(p, log)
    log(1.0, "Tabelas 2 e 3 e Fig. 7 (descritivas da amostra principal)")
    paper_tables.gerar(p)
    log(1.0, "XGBoost (robustez sem faixas)")
    t = xgb.tabela()
    t.to_csv(p / "robustez_xgboost.csv", index=False)
    (p / "tabela_robustez_xgboost.tex").write_text(xgb.tabela_tex(t))
    print("gravado em", p)
