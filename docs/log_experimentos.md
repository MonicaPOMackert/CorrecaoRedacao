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

### QWK por competência (bruto), Flash Lite

| Modo | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| holístico | 0,29 | 0,46 | 0,38 | 0,29 | 0,33 |
| MTS | 0,29 | 0,50 | 0,41 | 0,40 | 0,33 |
| MTS2 | 0,21 | 0,56 | 0,40 | 0,38 | 0,32 |

C1 (norma culta) e C5 (proposta de intervenção) são os gargalos. O MTS levanta C2, C3 e C4.
Nem o MTS nem o MTS2 mexeram em C1 e C5.

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

## Consumo por tipo de teste

- holístico: 1 chamada por redação
- MTS e MTS2: 5 chamadas por redação (uma por competência)
- self-consistency (ainda não usado): multiplica pelo número de amostras

## Próximos passos

1. C1 e C5 continuam o gargalo. Próxima ideia: few-shot com redações âncora (exemplo de
   proposta de intervenção parcial com a nota certa) em vez de mudar a estrutura do prompt.
   Mudar estrutura já foi testado no MTS2 e não funcionou.
2. Trilha de feedback formativo: o modo MTS já gera justificativa por competência no campo
   `raw`. Avaliar com rubrica mais avaliação humana, usando o Banco de Redações da UOL como
   referência parcial.
3. gpt-oss-120B em modo MTS (Groq), para ver se o ganho do MTS vale para o modelo aberto.
