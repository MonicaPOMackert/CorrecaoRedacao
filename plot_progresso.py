"""Grafico de progressao do QWK e QWK por competencia (holistico vs MTS).

Le os CSVs de predicao e RECALCULA tudo (reaproveita evaluate.py e calibrate.py),
nada de numero cravado a mao. Gera results/progresso_qwk.png.

Uso:
    python plot_progresso.py
"""
from __future__ import annotations

import contextlib
import io

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import calibrate  # noqa: E402
from evaluate import evaluate_pair  # noqa: E402

MAP = "results/v5_experimentos_v2/index_prompt_map.csv"
FL_H = "results/api/flashlite_holistico.csv"
FL_M = "results/api/flashlite_mts.csv"
GO_H = "results/api/groq_holistico.csv"


def qwk_total_bruto(path):
    d = pd.read_csv(path).dropna(subset=["score", "pred_total"])
    return evaluate_pair(np.asarray(d["score"], float), np.asarray(d["pred_total"], float),
                         is_total=True)["qwk"]


def qwk_total_calibrado(path):
    """QWK da nota total com deslocamento de vies, out-of-fold, folds por tema."""
    with contextlib.redirect_stdout(io.StringIO()):
        res = calibrate.run(path, prompts_file=MAP, out_csv=None, plot_path=None)
    r = res[(res["calibrador"] == "vies") & (res["tipo"] == "out-of-fold")]
    return float(r["qwk"].iloc[0])


def qwk_por_comp(path):
    d = pd.read_csv(path)
    out = {}
    for i in range(1, 6):
        s = d.dropna(subset=[f"c{i}", f"pred_c{i}"])
        out[f"C{i}"] = evaluate_pair(np.asarray(s[f"c{i}"], float),
                                     np.asarray(s[f"pred_c{i}"], float), is_total=False)["qwk"]
    return out


def main():
    prog = [
        ("Qwen 7B\nzero-shot\n(original)", qwk_total_bruto("results/v5_experimentos_v2/ck1_qwen.csv")),
        ("Gemma 7B\nfew-shot\n(calibrado)", qwk_total_calibrado("results/v5_experimentos_v2/ck2_gemma.csv")),
        ("gpt-oss-120B\nholistico\n(calibrado)", qwk_total_calibrado(GO_H)),
        ("Flash Lite\nholistico\n(calibrado)", qwk_total_calibrado(FL_H)),
        ("Flash Lite\nMTS\n(calibrado)", qwk_total_calibrado(FL_M)),
    ]
    comp_go, comp_h, comp_m = qwk_por_comp(GO_H), qwk_por_comp(FL_H), qwk_por_comp(FL_M)

    print("progressao QWK total:")
    for nome, v in prog:
        print(f"  {nome.replace(chr(10), ' ')}: {v:.3f}")
    print("por competencia (gpt-oss holis / FL holis / FL MTS):")
    for i in range(1, 6):
        print(f"  C{i}: {comp_go[f'C{i}']:.3f} / {comp_h[f'C{i}']:.3f} / {comp_m[f'C{i}']:.3f}")
    assert prog[-1][1] > prog[0][1], "MTS calibrado deveria superar o baseline"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels = [p[0] for p in prog]
    vals = [p[1] for p in prog]
    bars = ax1.bar(labels, vals, color=["#b0b0b0", "#9FB4DC", "#7594CB", "#5C7FBF", "#2E5EA8"])
    ax1.axhspan(0.60, 0.73, color="green", alpha=0.08)
    ax1.text(4.35, 0.665, "faixa\nliteratura", ha="center", fontsize=8, color="green")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=11)
    ax1.set_ylabel("QWK (nota total, cross-prompt)")
    ax1.set_ylim(0, 0.8)
    ax1.set_title("Progressao do QWK")

    x = np.arange(5)
    w = 0.27
    ax2.bar(x - w, [comp_go[f"C{i}"] for i in range(1, 6)], w, label="gpt-oss-120B holis.", color="#9FB4DC")
    ax2.bar(x, [comp_h[f"C{i}"] for i in range(1, 6)], w, label="Flash Lite holis.", color="#5C7FBF")
    ax2.bar(x + w, [comp_m[f"C{i}"] for i in range(1, 6)], w, label="Flash Lite MTS", color="#2E5EA8")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"C{i}" for i in range(1, 6)])
    ax2.set_ylabel("QWK por competencia (bruto)")
    ax2.set_ylim(0, 0.6)
    ax2.set_title("QWK por competencia")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("results/progresso_qwk.png", dpi=130)
    print("\nsalvo em results/progresso_qwk.png")


if __name__ == "__main__":
    main()
