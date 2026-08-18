# Correção Automática de Redações do ENEM com LLMs

> Iniciação Científica (UTFPR) — correção automática de redações do ENEM usando Modelos de Linguagem Pré-Treinados (LLMs), com geração de *feedback* formativo para o aluno.

## Sobre o projeto

O ENEM avalia redações em 5 competências (C1–C5), cada uma pontuada de 0 a 200. Fazer essa correção manualmente é caro e lento em escala — este projeto investiga o quão bem LLMs abertos conseguem reproduzir essa nota, e, mais importante, gerar um **feedback formativo** (não só uma nota, mas uma explicação do que o aluno pode melhorar).

O projeto testa diferentes modelos (Qwen, Llama, Mistral, Gemma2, em tamanhos de ~7B até 70B/72B parâmetros) e diferentes estratégias de *prompting* e *fine-tuning*, comparando a nota gerada pelo modelo com a nota humana de referência.

## Evolução dos experimentos

O projeto foi construído de forma incremental, cada etapa usando o aprendizado da anterior:

| Etapa | O que testa | Pasta |
|---|---|---|
| v1 — Zero-shot | Modelos avaliando redação sem exemplo prévio | [`notebooks/v1_zero_shot`](notebooks/v1_zero_shot) |
| v2 — Few-shot | Mesmos modelos, agora com exemplos no prompt | [`notebooks/v2_few_shot`](notebooks/v2_few_shot) |
| v3 — Fine-tuning (LoRA) | Ajuste fino dos modelos no dataset de redações via LoRA | [`notebooks/v3_finetuning`](notebooks/v3_finetuning) |
| v4 — CoT + Instruction Tuning | *Chain-of-thought* e ajuste por instrução, geração de feedback formativo | [`notebooks/v4_cot_instruction_tuning`](notebooks/v4_cot_instruction_tuning) |
| v5 — Experimentos V2.0 | Re-execução consolidada de zero/few-shot com mais modelos (checkpoints por fold) | [`notebooks/v5_experimentos_v2`](notebooks/v5_experimentos_v2) |
| v6 — Experimentos V3.0 | Escala para modelos maiores (Llama 70B, Qwen 72B) | [`notebooks/v6_experimentos_v3`](notebooks/v6_experimentos_v3) |

### Status atual

A etapa v6 (modelos de 70B/72B) foi interrompida no meio da execução por falta de crédito computacional no Google Colab Pro — por isso inclui uma tentativa adicional com vLLM (`exp_all_qwen72b_vllm_v3.ipynb`) como alternativa mais eficiente de inferência. Os notebooks dessa etapa refletem o estado real em que os experimentos pararam, não uma versão "limpa" — optei por manter assim para documentar o processo real de pesquisa, não só o resultado final.

## Metodologia

- **Dataset:** redações do ENEM anonimizadas (dataset [essay-br](https://github.com/lplnufpi/essay-br)), com nota de referência por competência (C1–C5) e nota final.
- **Validação:** k-fold cross-validation para avaliar consistência entre folds.
- **Ajuste de hiperparâmetros:** Optuna, para busca automática de configurações de fine-tuning.
- **Avaliação:** comparação nota-a-nota com referência humana, e BERTScore para qualidade textual do feedback gerado.
- **Fine-tuning:** LoRA (Low-Rank Adaptation) para ajuste eficiente dos modelos sem re-treinar todos os parâmetros.

## Tecnologias

Python · Jupyter/Google Colab · Hugging Face Transformers · PEFT (LoRA) · Optuna · BERTScore · Qwen 2.5 · Llama 3 · Mistral · Gemma 2

## Adaptadores LoRA (pesos dos modelos)

Os pesos dos modelos ajustados (LoRA adapters) não estão neste repositório por tamanho (200MB+ ao todo) — a prática recomendada é publicá-los no Hugging Face Hub. As configurações usadas em cada fine-tuning (hiperparâmetros do LoRA) estão documentadas em [`docs/lora_configs`](docs/lora_configs).

Adaptadores publicados no Hugging Face Hub:
- [Qwen 2.5 — qwen-essay-scorer-lora](https://huggingface.co/YurinhoMatsumoto/qwen-essay-scorer-lora)
- [Llama 3.1 — llama-essay-scorer-lora](https://huggingface.co/YurinhoMatsumoto/llama-essay-scorer-lora)
- [Mistral — mistral-essay-scorer-lora](https://huggingface.co/YurinhoMatsumoto/mistral-essay-scorer-lora)
- [Gemma 2 — gemma2-essay-scorer-lora](https://huggingface.co/YurinhoMatsumoto/gemma2-essay-scorer-lora)

## Estrutura do repositório

```
notebooks/   → um notebook por etapa do experimento (v1 a v6)
results/     → gráficos e CSVs de resultado de cada etapa
data/        → dataset usado nos experimentos de feedback (v4)
docs/        → configurações dos adaptadores LoRA e referências bibliográficas
```

## Colaboração

Projeto desenvolvido por Yuri Matsumoto, com colaboração de Mônica em parte dos experimentos de CoT/instruction tuning (v4).

## Referências

Ver [`docs/referencias.md`](docs/referencias.md) para a lista de artigos que embasaram a metodologia.

---

# Automated ENEM Essay Scoring with LLMs

> Undergraduate research project (UTFPR) — automated scoring of Brazilian ENEM exam essays using Pre-trained Language Models (LLMs), with formative feedback generation for students.

## About the project

The ENEM exam scores essays across 5 competencies (C1–C5), each rated 0–200. Manual grading at scale is slow and expensive — this project investigates how well open LLMs can reproduce that score and, more importantly, generate **formative feedback** (not just a grade, but an explanation of what the student can improve).

The project evaluates multiple models (Qwen, Llama, Mistral, Gemma2, ranging from ~7B to 70B/72B parameters) and multiple prompting/fine-tuning strategies, comparing model-generated scores against human reference scores.

## Experiment evolution

| Stage | What it tests | Folder |
|---|---|---|
| v1 — Zero-shot | Models scoring essays with no prior example | [`notebooks/v1_zero_shot`](notebooks/v1_zero_shot) |
| v2 — Few-shot | Same models, now with examples in the prompt | [`notebooks/v2_few_shot`](notebooks/v2_few_shot) |
| v3 — Fine-tuning (LoRA) | Fine-tuning models on the essay dataset via LoRA | [`notebooks/v3_finetuning`](notebooks/v3_finetuning) |
| v4 — CoT + Instruction Tuning | Chain-of-thought and instruction tuning, formative feedback generation | [`notebooks/v4_cot_instruction_tuning`](notebooks/v4_cot_instruction_tuning) |
| v5 — Experiments V2.0 | Consolidated re-run of zero/few-shot with more models (per-fold checkpoints) | [`notebooks/v5_experimentos_v2`](notebooks/v5_experimentos_v2) |
| v6 — Experiments V3.0 | Scaling up to larger models (Llama 70B, Qwen 72B) | [`notebooks/v6_experimentos_v3`](notebooks/v6_experimentos_v3) |

### Current status

Stage v6 (70B/72B models) was interrupted mid-run when Google Colab Pro compute credits ran out — which is why it also includes an additional attempt using vLLM (`exp_all_qwen72b_vllm_v3.ipynb`) as a more efficient inference alternative. Notebooks in this stage reflect the actual state the experiments stopped at, not a "cleaned up" version — kept this way intentionally to document the real research process, not just the final result.

## Methodology

- **Dataset:** anonymized ENEM essays ([essay-br](https://github.com/lplnufpi/essay-br) dataset), with reference scores per competency (C1–C5) and final score.
- **Validation:** k-fold cross-validation to assess consistency across folds.
- **Hyperparameter tuning:** Optuna, for automated fine-tuning configuration search.
- **Evaluation:** score-by-score comparison against human reference, and BERTScore for generated feedback text quality.
- **Fine-tuning:** LoRA (Low-Rank Adaptation) for efficient adaptation without retraining all parameters.

## Tech stack

Python · Jupyter/Google Colab · Hugging Face Transformers · PEFT (LoRA) · Optuna · BERTScore · Qwen 2.5 · Llama 3 · Mistral · Gemma 2

## LoRA adapters (model weights)

Fine-tuned model weights (LoRA adapters) are not included in this repository due to size (200MB+ combined) — the recommended practice is publishing them on the Hugging Face Hub instead. The configuration used for each fine-tuning run (LoRA hyperparameters) is documented under [`docs/lora_configs`](docs/lora_configs).

Adapters published on the Hugging Face Hub:
- [Qwen 2.5 — qwen-essay-scorer-lora](https://huggingface.co/YurinhoMatsumoto/qwen-essay-scorer-lora)
- [Llama 3.1 — llama-essay-scorer-lora](https://huggingface.co/YurinhoMatsumoto/llama-essay-scorer-lora)
- [Mistral — mistral-essay-scorer-lora](https://huggingface.co/YurinhoMatsumoto/mistral-essay-scorer-lora)
- [Gemma 2 — gemma2-essay-scorer-lora](https://huggingface.co/YurinhoMatsumoto/gemma2-essay-scorer-lora)

## Repository structure

```
notebooks/   → one notebook per experiment stage (v1 to v6)
results/     → charts and result CSVs for each stage
data/        → dataset used in the feedback experiments (v4)
docs/        → LoRA adapter configs and bibliography
```

## Collaboration

Developed by Yuri Matsumoto, with Mônica collaborating on part of the CoT/instruction tuning experiments (v4).

## References

See [`docs/referencias.md`](docs/referencias.md) for the list of papers that informed the methodology.
