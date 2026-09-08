"""Confere as metricas do evaluate.py contra scikit-learn / scipy e contra
um numero ja calculado pelo notebook v5. Se tudo bater, as metricas estao certas.

    python verify_metrics.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from evaluate import _to_bands, evaluate_pair, qwk


def main():
    rng = np.random.default_rng(0)
    falhas = 0

    # 1. QWK hand-rolled vs sklearn, em varios cenarios sinteticos
    for _ in range(200):
        n = rng.integers(30, 400)
        gold = rng.choice(np.arange(0, 1001, 40), size=n)
        pred = np.clip(gold + rng.normal(0, rng.uniform(20, 300), n), 0, 1000)
        gb, pb = _to_bands(gold, 40, 25), _to_bands(pred, 40, 25)
        meu = qwk(gb, pb, 26)
        ref = cohen_kappa_score(gb, pb, weights="quadratic", labels=list(range(26)))
        if not np.isclose(meu, ref, atol=1e-9):
            print(f"  DIVERGE QWK: meu={meu:.6f} sklearn={ref:.6f}")
            falhas += 1
    print(f"1. QWK vs sklearn (200 casos): {'OK' if falhas == 0 else str(falhas) + ' falhas'}")

    # 2. Pearson / Spearman vs scipy num CSV real
    d = pd.read_csv("results/api/flashlite_mts.csv").dropna(subset=["score", "pred_total"])
    g = np.asarray(d["score"], float)
    p = np.asarray(d["pred_total"], float)
    m = evaluate_pair(g, p, is_total=True)
    checks = {
        "pearson": (m["pearson"], pearsonr(g, p).statistic),
        "spearman": (m["spearman"], spearmanr(g, p).statistic),
        "mae": (m["mae"], float(np.mean(np.abs(p - g)))),
        "qwk": (m["qwk"], cohen_kappa_score(_to_bands(g, 40, 25), _to_bands(p, 40, 25),
                                            weights="quadratic", labels=list(range(26)))),
    }
    for nome, (meu, ref) in checks.items():
        ok = np.isclose(meu, ref, atol=1e-6)
        falhas += not ok
        print(f"2. {nome:9s} meu={meu:.6f}  ref={ref:.6f}  {'OK' if ok else 'DIVERGE'}")

    # 3. Cruzar com o numero que o notebook v5 ja calculou
    #    resultados_exp1_exp2_final.csv linha "Qwen 2.5 7B (Zero-Shot)" foi feita
    #    pelo notebook (sklearn) sobre os mesmos dados de ck1_qwen.csv
    ck = pd.read_csv("results/v5_experimentos_v2/ck1_qwen.csv").dropna(subset=["pred_total"])
    meu_qwk = evaluate_pair(np.asarray(ck["score"], float),
                            np.asarray(ck["pred_total"], float), is_total=True)["qwk"]
    ref_row = pd.read_csv("results/v5_experimentos_v2/resultados_exp1_exp2_final.csv")
    ref_qwk = float(ref_row.loc[ref_row["modelo"] == "Qwen 2.5 7B (Zero-Shot)", "qwk"].iloc[0])
    ok = np.isclose(meu_qwk, ref_qwk, atol=1e-3)
    falhas += not ok
    print(f"3. QWK ck1_qwen: evaluate.py={meu_qwk:.4f}  notebook v5={ref_qwk:.4f}  "
          f"{'OK' if ok else 'DIVERGE'}")

    print(f"\n{'TUDO BATE' if falhas == 0 else str(falhas) + ' DIVERGENCIAS'}")


if __name__ == "__main__":
    main()
