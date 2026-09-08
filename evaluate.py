"""Avaliacao padronizada de predicoes de nota de redacao (ENEM / essay-br).

Recebe um ou mais CSVs de predicao e devolve, por alvo (nota total e cada
competencia C1..C5 quando presentes), o pacote de metricas:

    n, n_dropado, MAE, RMSE, Pearson, Spearman, QWK (quadratic weighted kappa),
    acuracia exata por faixa, acuracia adjacente (+/- 1 faixa), vies (media do
    erro com sinal), media humana e media do modelo.

Colunas reconhecidas no CSV de entrada (deteccao automatica):
    nota total humana : score  (ou gold, score_total, nota)
    nota total modelo : pred_total  (ou pred, pred_score, nota_pred)
    competencia humana: c1..c5
    competencia modelo: pred_c1..pred_c5  (ou c1_pred..c5_pred)

Faixas para QWK e acuracia: competencia arredondada para multiplo de 40
(classes 0..5); total arredondado para multiplo de 40 (classes 0..25).

Uso:
    python evaluate.py results/v5_experimentos_v2/ck1_qwen.csv ... --out consolidado.csv
    python evaluate.py results/**/resultados_*.csv --out consolidado.csv --confusao
    python evaluate.py            # roda o auto-teste

Sem dependencias alem de pandas + numpy.
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import pandas as pd

GOLD_TOTAL = ["score", "gold", "score_total", "nota", "nota_humana", "y_true"]
PRED_TOTAL = ["pred_total", "pred", "pred_score", "nota_pred", "y_pred", "predicao"]


def _first_present(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _detect_targets(df):
    """Devolve lista de (nome_alvo, col_gold, col_pred) presentes no df."""
    cols = list(df.columns)
    targets = []
    gt = _first_present(cols, GOLD_TOTAL)
    pt = _first_present(cols, PRED_TOTAL)
    if gt and pt:
        targets.append(("total", gt, pt))
    for i in range(1, 6):
        g = _first_present(cols, [f"c{i}"])
        p = _first_present(cols, [f"pred_c{i}", f"c{i}_pred", f"pred{i}"])
        if g and p:
            targets.append((f"c{i}", g, p))
    return targets


def _to_bands(x, step=40, hi=5):
    """Score -> classe ordinal (multiplo de step), recortada em [0, hi]."""
    b = np.round(np.asarray(x, dtype=float) / step)
    return np.clip(b, 0, hi).astype(int)


def qwk(gold_bands, pred_bands, n_classes):
    """Quadratic weighted kappa entre dois vetores de classes ordinais 0..n_classes-1."""
    g = np.asarray(gold_bands, dtype=int)
    p = np.asarray(pred_bands, dtype=int)
    O = np.zeros((n_classes, n_classes), dtype=float)
    for a, b in zip(g, p):
        O[a, b] += 1
    total = O.sum()
    if total == 0:
        return float("nan")
    idx = np.arange(n_classes)
    W = (idx[:, None] - idx[None, :]) ** 2 / max((n_classes - 1) ** 2, 1)
    E = np.outer(O.sum(axis=1), O.sum(axis=0)) / total
    denom = (W * E).sum()
    if denom == 0:  # sem variacao esperada -> concordancia perfeita trivial
        return 1.0
    return 1.0 - (W * O).sum() / denom


def _pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(x):
    """Ranks medios (trata empates), equivalente a scipy rankdata 'average'."""
    x = np.asarray(x, dtype=float)
    order = x.argsort()
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # media dos ranks em grupos de empate
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return sums[inv] / counts[inv]


def _spearman(a, b):
    return _pearson(_rankdata(a), _rankdata(b))


def evaluate_pair(gold, pred, is_total):
    """Metricas para um alvo. gold/pred: arrays de score bruto ja limpos (sem NaN)."""
    gold = np.asarray(gold, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - gold
    hi = 25 if is_total else 5
    gb, pb = _to_bands(gold, 40, hi), _to_bands(pred, 40, hi)
    band_diff = np.abs(gb - pb)
    return {
        "n": len(gold),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "pearson": _pearson(gold, pred),
        "spearman": _spearman(gold, pred),
        "qwk": qwk(gb, pb, hi + 1),
        "acc_exata": float(np.mean(band_diff == 0)),
        "acc_adjacente": float(np.mean(band_diff <= 1)),
        "vies": float(np.mean(err)),
        "media_humano": float(np.mean(gold)),
        "media_modelo": float(np.mean(pred)),
    }


def evaluate_file(path, show_confusion=False):
    df = pd.read_csv(path)
    targets = _detect_targets(df)
    if not targets:
        print(f"[aviso] {path}: nenhuma coluna de nota reconhecida, pulando.", file=sys.stderr)
        return []
    rows = []
    for name, gcol, pcol in targets:
        sub = df[[gcol, pcol]].apply(pd.to_numeric, errors="coerce")
        n_raw = len(sub)
        sub = sub.dropna()
        if sub.empty:
            print(f"[aviso] {path} [{name}]: sem linhas validas.", file=sys.stderr)
            continue
        m = evaluate_pair(np.asarray(sub[gcol], float), np.asarray(sub[pcol], float),
                          is_total=(name == "total"))
        m = {"arquivo": path, "alvo": name, "n_dropado": n_raw - m["n"], **m}
        rows.append(m)
        if show_confusion:
            hi = 25 if name == "total" else 5
            gb = _to_bands(np.asarray(sub[gcol], float), 40, hi)
            pb = _to_bands(np.asarray(sub[pcol], float), 40, hi)
            cm = pd.crosstab(pd.Series(gb, name="humano"), pd.Series(pb, name="modelo"))
            print(f"\n=== matriz de confusao (faixas) {path} [{name}] ===")
            print(cm.to_string())
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Avaliacao padronizada de predicoes de nota de redacao.")
    ap.add_argument("files", nargs="*", help="CSVs de predicao (aceita glob)")
    ap.add_argument("--out", help="CSV de saida com a tabela de metricas")
    ap.add_argument("--confusao", action="store_true", help="imprime matriz de confusao por faixa")
    args = ap.parse_args(argv)

    if not args.files:
        demo()
        return

    paths = []
    for pat in args.files:
        hit = glob.glob(pat, recursive=True)
        paths.extend(hit or [pat])

    rows = []
    for p in paths:
        rows.extend(evaluate_file(p, show_confusion=args.confusao))
    if not rows:
        print("nenhuma metrica gerada.", file=sys.stderr)
        sys.exit(1)

    out = pd.DataFrame(rows)
    cols = ["arquivo", "alvo", "n", "n_dropado", "mae", "rmse", "pearson", "spearman",
            "qwk", "acc_exata", "acc_adjacente", "vies", "media_humano", "media_modelo"]
    out = out[cols]
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(out.to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nsalvo em {args.out}")


def demo():
    """Auto-teste: predicao perfeita -> metricas ideais; ruido -> QWK cai."""
    rng = np.random.default_rng(0)
    gold = rng.choice([0, 200, 400, 600, 800, 1000], size=400)

    perf = evaluate_pair(gold, gold, is_total=True)
    assert perf["mae"] == 0 and perf["rmse"] == 0, perf
    assert abs(perf["qwk"] - 1.0) < 1e-9, perf
    assert abs(perf["pearson"] - 1.0) < 1e-9, perf
    assert perf["acc_exata"] == 1.0 and perf["acc_adjacente"] == 1.0, perf
    assert abs(perf["vies"]) < 1e-9, perf

    noisy = evaluate_pair(gold, gold + rng.normal(0, 120, size=gold.shape), is_total=True)
    assert 0.0 < noisy["qwk"] < perf["qwk"], noisy
    assert noisy["mae"] > 0, noisy

    biased = evaluate_pair(gold, np.clip(gold + 200, 0, 1000), is_total=True)
    assert biased["vies"] > 0, biased

    # QWK conhecido: 2 classes, meia concordancia
    g = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 0, 1])
    assert abs(qwk(g, p, 2) - 0.0) < 1e-9, qwk(g, p, 2)

    # deteccao de colunas
    df = pd.DataFrame({"score": [100], "pred_total": [120], "c1": [20], "pred_c1": [40]})
    tg = _detect_targets(df)
    assert ("total", "score", "pred_total") in tg and ("c1", "c1", "pred_c1") in tg, tg

    print("demo ok: metricas ideais para predicao perfeita, QWK cai com ruido, vies detecta deslocamento.")


if __name__ == "__main__":
    main()
