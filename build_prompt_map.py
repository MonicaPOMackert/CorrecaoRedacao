"""Reconstroi o mapa index_redacao -> prompt (tema) do conjunto de teste do v5.

O v5 (notebooks/v5_experimentos_v2) monta o teste assim (celula 3):
    df = read_csv('meu_dataset.csv')
    df = df[df['score'] > 0].reset_index(drop=True)
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, df['score'])
    df_teste = df[fold == 0].reset_index(drop=True)
e os checkpoints ck*.csv usam index_redacao = posicao da linha em df_teste.

Semente fixa, entao da para reproduzir. O script valida comparando a coluna
`score` reconstruida com a de um checkpoint antes de salvar o mapa.

Uso:
    python build_prompt_map.py            # usa caminhos padrao do repo
    python build_prompt_map.py --dataset data/meu_dataset.csv \
        --check results/v5_experimentos_v2/ck1_qwen.csv \
        --out results/v5_experimentos_v2/index_prompt_map.csv
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def build_test_df(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df[df["score"] > 0].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df["fold"] = -1
    for i, (_, te) in enumerate(skf.split(df, df["score"])):
        df.loc[te, "fold"] = i
    return df[df["fold"] == 0].reset_index(drop=True)


def validate(df_teste, check_path):
    ck = pd.read_csv(check_path)
    merged = ck.merge(
        df_teste[["prompt", "score"]].rename(columns={"score": "score_reconstruido"}),
        left_on="index_redacao", right_index=True, how="left",
    )
    bad = merged["score"].ne(merged["score_reconstruido"]).sum()
    n = len(merged)
    print(f"validacao contra {check_path}: {n - bad}/{n} linhas com score batendo")
    return bad == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/meu_dataset.csv")
    ap.add_argument("--check", default="results/v5_experimentos_v2/ck1_qwen.csv")
    ap.add_argument("--out", default="results/v5_experimentos_v2/index_prompt_map.csv")
    args = ap.parse_args(argv)

    df_teste = build_test_df(args.dataset)
    print(f"df_teste reconstruido: {len(df_teste)} redacoes, {df_teste['prompt'].nunique()} temas")

    if not validate(df_teste, args.check):
        print("ERRO: score reconstruido nao bate com o checkpoint. Split divergente "
              "(versao do sklearn?). Nao vou salvar o mapa.", file=sys.stderr)
        sys.exit(1)

    out = df_teste.reset_index().rename(columns={"index": "index_redacao"})[["index_redacao", "prompt"]]
    out.to_csv(args.out, index=False)
    print(f"mapa salvo em {args.out}  ({len(out)} linhas)")


if __name__ == "__main__":
    main()
