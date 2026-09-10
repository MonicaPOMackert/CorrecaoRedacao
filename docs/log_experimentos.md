# Log de experimentos, reteste com APIs gratuitas

Registro datado de cada teste e do estado das cotas diárias das APIs gratuitas.
Objetivo: saber a qualquer momento o que já foi rodado, com que resultado, e quanta
cota resta no dia.

Todos os QWK são na nota total, avaliação cross-prompt (folds por tema), 300 redações
do conjunto de teste do v5 salvo em `data/amostra_300.csv`, salvo indicação de `n`
menor. "Calibrado" = deslocamento de viés aprendido out-of-fold (`calibrate.py`).

## Resultados por data

| Data | Modelo | Modo | n | QWK bruto | QWK calibrado | Pearson | Observação |
|---|---|---|---|---|---|---|---|
| 2026-09-02 | Qwen 2.5 7B | zero-shot | 1293 | 0,35 | 0,35 | 0,37 | melhor resultado anterior do projeto, recalculado |
| 2026-09-02 | Gemma 2 9B | few-shot | 1297 | 0,25 | 0,42 | 0,44 | melhor 7B após calibração. Achado: erro é viés de escala, não discriminação |
| 2026-09-03 | Gemini 3.5 Flash Lite | holístico | 300 | 0,47 | 0,54 | 0,55 | primeiro resultado por API a bater a linha de base |
| 2026-09-03 | Gemini 3.5 Flash Lite | MTS | 298 | 0,53 | 0,60 | 0,60 | melhor resultado. 0,60 é o piso da literatura para essay-br |
| 2026-09-03 | gpt-oss-120B (Groq) | holístico | 300 | 0,23 | 0,50 | 0,55 | viés bruto -226. Discrimina igual ao Flash Lite, escala muito pior |
| 2026-09-03 | Gemini 3.8 Flash | holístico | 60 | 0,40 | 0,42 | 0,42 | modelo preview, RPD real ~13 por conta. Pior que o Flash Lite. Descartado |
| 2026-09-04 | Gemini 3.5 Flash Lite | MTS2 (C1/C5 dedicado) | 257 | 0,54 | 0,60 | 0,60 | prompt estruturado para C1 e C5. Não melhorou. C1 piorou (0,29 para 0,21), C5 igual. Descartado |
| 2026-09-04 | verificação de métricas | - | - | - | - | - | `verify_metrics.py`: QWK, Pearson, Spearman e MAE batem com scikit-learn/scipy e com o valor calculado pelo notebook v5 |
| 2026-09-08 | Gemini 3.5 Flash Lite | MTS_FS (few-shot com âncoras) | 297 | 0,58 | 0,60 | 0,62 | uma redação âncora por faixa de nota por competência. Sobe a discriminação bruta (Pearson 0,60 para 0,62) e reduz o viés (-82 para -59), mas o total calibrado empata em 0,60. C1 0,29 para 0,34, C3 e C4 sobem, C5 travado em 0,33 |
| 2026-09-09 | Gemini 3.5 Flash Lite | MTS_FS2 (âncoras + checklist no C5) | 296 | 0,59 | 0,61 | 0,60 | âncoras em C1 a C4, mais checklist dos 5 elementos e aviso "parcial não é 0" só no C5. C5 sai de 0,33 para 0,35 e o viés do C5 quase some (média 90 para 113, humano 119). Melhor config até agora: QWK bruto 0,59, viés geral -36. Total calibrado 0,605. Adotado como padrão |

### QWK por competência (bruto), Flash Lite

| Modo | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| holístico | 0,29 | 0,46 | 0,38 | 0,29 | 0,33 |
| MTS | 0,29 | 0,50 | 0,41 | 0,40 | 0,33 |
| MTS2 | 0,21 | 0,56 | 0,40 | 0,38 | 0,32 |
| MTS_FS | 0,34 | 0,48 | 0,46 | 0,49 | 0,33 |
| MTS_FS2 | 0,35 | 0,47 | 0,43 | 0,48 | 0,35 |

O few-shot com âncoras (MTS_FS) move C1 (0,29 para 0,34) e sobe C3 e C4. O MTS_FS2 adiciona
checklist no C5 e finalmente tira o C5 do lugar (0,33 para 0,35), cortando o viés de nota 0.
C5 segue sendo o pior trait. Próximo alvo: Reflect-and-Revise da rubrica de C5.

## Cotas diárias das APIs gratuitas

Sempre conferir antes de rodar, os provedores mudam os limites com frequência.

### Gemini (Google AI Studio), limite por projeto Google

| Modelo | RPM | RPD | TPM | Redações/dia por conta |
|---|---|---|---|---|
| gemini-3.5-flash-lite | 15 | 500 | 250K | holístico ~500, MTS ~100 |
| gemini-3.x-flash (cheio) | 5 | ~20 (real ~13) | 250K | ~13 a 20 |
| gemma-4-31b-it | 30 | 14.400 | 16K | limitado pelo TPM, ~5/min |

Reset do RPD: meia-noite horário do Pacífico, cerca de 04:00 no horário de Brasília.
Temos 5 contas Google, então multiplicar por 5 (usar `--key-env GEMINI_API_KEY_1..5` e `--offset`).

### Groq, limite por conta

| Modelo | RPM | RPD | TPM | TPD | Redações/dia por conta |
|---|---|---|---|---|---|
| openai/gpt-oss-120b | 30 | 1.000 | 8K | 200K | holístico ~65 (limitado pelo TPD) |
| qwen/qwen3.6-27b | 30 | 1.000 | 8K | 200K | ~65 |

Temos 5 contas Groq. gpt-oss-120b holístico nas 300: cabe em 1 dia com as 5 contas.

### Fora

- Cerebras: sem crédito, trial de 5 dólares não foi provisionado para a organização.
- Maritaca (Sabiá): pedido de créditos acadêmicos feito no início de 2026, sem retorno.

## Gasto diário registrado

| Data | O que rodou | Chamadas por conta Gemini | Sobra estimada |
|---|---|---|---|
| 2026-09-08 | mts_fs 300 (60 x 5) + 2 smokes | ~350 (conta 1), ~300 (contas 2-5) | conta 1 ~150, contas 2-5 ~200 |
| 2026-09-09 | mts_fs2 300 (60 x 5), rodado pela colega na máquina dela | ~300 por conta | ~200 por conta |

Regra: antes de disparar um run grande, somar o que já foi gasto no dia. mts_fs / mts_fs2
de 300 redações custam 300 chamadas por conta (60 redações x 5 competências). Cabe 1 run
por dia por lote de 5 contas. RPD zera ~04:00 BRT.

## Consumo por tipo de teste

- holístico: 1 chamada por redação
- MTS e MTS2: 5 chamadas por redação (uma por competência)
- self-consistency (ainda não usado): multiplica pelo número de amostras

## Próximos passos

1. **C5 é o gargalo isolado.** MTS_FS (âncoras) resolveu C1 mas não C5. Próximo teste
   (`--modo mts_fs2`): combinar âncoras + checklist dos 5 elementos só no C5, com aviso
   explícito de que nota parcial não é 0. Código já pronto.
2. Se o mts_fs2 também não resolver C5: Reflect-and-Revise da rubrica de C5 (CoNLL 2026),
   iterativo num fold de validação com gabarito.
3. Features linguísticas para C1: rodar LanguageTool (offline, pt-BR) e injetar a contagem
   de erros no prompt de C1 como contexto.
4. Trilha de feedback formativo: o modo MTS já gera justificativa por competência no campo
   `raw`. Avaliar com rubrica mais avaliação humana, usando o Banco de Redações da UOL como
   referência parcial.
5. gpt-oss-120B em modo MTS (Groq), para ver se o ganho do MTS vale para o modelo aberto.

### Base na literatura (Qualis A/A1)

- Few-shot com âncoras: "Anchor is the key" (Studies in Educational Evaluation, 2026);
  "Specialists or Generalists?" (2026) reporta +26% de QWK com 2 exemplos por faixa.
- Reflect-and-Revise: "Automated Refinement of Essay Scoring Rubrics via Reflect-and-Revise"
  (CoNLL 2026), ganho de até +0,4 QWK sem treino.
- Features linguísticas: "Improve LLM-based AES with Linguistic Features" (arXiv 2502.09497).
- Teto realista: "Has AES Reached Sufficient Accuracy? QWK Ceilings from Classical Test
  Theory" (arXiv 2604.19131) e PROPOR 2026 (gap ao oráculo de 0,15 a 0,29 por trait).
  Alvo realista para o total: 0,65 a 0,68, não 0,73.
