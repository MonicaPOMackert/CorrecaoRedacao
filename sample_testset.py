"""Amostra fixa e estratificada do conjunto de teste do v5, para os retestes por API.

Reconstroi o mesmo df_teste do v5 (ver build_prompt_map.py), depois sorteia N
redacoes com semente fixa, estratificando por faixa de nota final (0-1000 em
passos de 200) e cobrindo o maximo de temas possivel.

Saida: CSV com index_redacao (posicao no df_teste do v5, casa com os ck*.csv),
prompt (tema), essay, c1..c5, score.

Uso:
    python sample_testset.py                 # 300 redacoes -> data/amostra_300.csv
    python sample_testset.py --n 500 --out data/amostra_500.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from build_prompt_map import build_test_df

SEED = 20260902


def stratified_sample(df, n, seed=SEED):
    """Amostra n linhas, proporcional por faixa de nota, espalhando entre temas."""
    faixa = np.clip((df["score"] // 200).astype(int), 0, 4)  # 0-199, ..., 800-1000
    picks = []
    for f, grp in df.groupby(faixa):
        k = round(n * len(grp) / len(df))
        grp = grp.sample(frac=1, random_state=seed + int(f))  # embaralha
        # ordena para variar o tema entre escolhas consecutivas
        grp = grp.sort_values("prompt", kind="stable")
        step = max(len(grp) // max(k, 1), 1)
        idx = list(range(0, len(grp), step))[:k]
        picks.append(grp.iloc[idx])
    out = pd.concat(picks)
    # acerta o total exato (arredondamentos podem sobrar/faltar 1-2)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    elif len(out) < n:
        resto = df.drop(out.index).sample(n=n - len(out), random_state=seed)
        out = pd.concat([out, resto])
    return out.sort_index()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/meu_dataset.csv")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/amostra_300.csv")
    args = ap.parse_args(argv)

    df = build_test_df(args.dataset).reset_index(names="index_redacao")
    amostra = stratified_sample(df, args.n)

    cols = ["index_redacao", "prompt", "title", "essay", "c1", "c2", "c3", "c4", "c5", "score"]
    amostra[cols].to_csv(args.out, index=False)

    print(f"{len(amostra)} redacoes -> {args.out}")
    print(f"temas cobertos: {amostra['prompt'].nunique()} de {df['prompt'].nunique()}")
    print("distribuicao por faixa de nota (200 pts):")
    faixa = (amostra["score"] // 200).astype(int).map(
        {0: "0-199", 1: "200-399", 2: "400-599", 3: "600-799", 4: "800-1000"})
    print(faixa.value_counts().sort_index().to_string())
    print(f"media da amostra: {amostra['score'].mean():.1f} | "
          f"media do teste completo: {df['score'].mean():.1f}")


if __name__ == "__main__":
    main()
