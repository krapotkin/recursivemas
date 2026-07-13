# RecursiveMAS — Изучение и эксперименты

> **Recursive Multi-Agent Systems** — фреймворк для масштабирования коллаборации агентов через рекурсию в латентном пространстве.

**Статья:** [arXiv:2604.25917](https://arxiv.org/abs/2604.25917)
**Репозиторий:** [RecursiveMAS/RecursiveMAS](https://github.com/RecursiveMAS/RecursiveMAS)
**Модели:** [HuggingFace/RecursiveMAS](https://huggingface.co/RecursiveMAS/models)
**Демо:** [Playground](https://vishalmysore.github.io/recursiveMASDemo/)

---

## Что такое RecursiveMAS?

Обычные мульти-агентные системы (MAS) передают информацию между агентами через **текст**: агент A генерирует текст, агент B читает текст, и так далее. Это дорого (много токенов), медленно (каждый агент декодирует текст) и плохо обучается (градиенты затухают через текстовые границы).

**RecursiveMAS** заменяет текстовую коммуникацию на **латентную**: агенты обмениваются векторными представлениями (embeddings) напрямую, минуя декодирование в текст. Ключевые компоненты:

### RecursiveLink

- **Inner Link** — внутри каждого агента: преобразует last-layer hidden state обратно в input embedding space, позволяя агенту генерировать "латентные мысли" без текстового декодирования
- **Outer Link** — между агентами: проецирует латентные представления одного агента в пространство другого, даже если модели разной архитектуры и размера

### Inner-Outer Loop Training

1. **Inner Loop** — каждый агент обучается генерировать латентные мысли, близкие к embedding'ам ground-truth ответа (косинусная регрессия)
2. **Outer Loop** — вся система обучается как единое целое: градиенты проходят через все рекурсивные итерации, оптимизируя только Outer Link модули

### Результаты (из статьи)

- **+8.3%** средняя точность vs лучшие базовые методы
- **1.2x–2.4x** ускорение инференса
- **34.6%–75.6%** сокращение использования токенов

---

## Структура проекта

```
recursivemas/
├── README.md        ← Этот файл
├── AGENTS.md        ← Контекст для агента
├── INSTALL.md       ← Инструкция по развёртыванию
├── theory/          ← Теоретические заметки и конспекты
├── experiments/     ← Воспроизведение результатов из статьи
│   └── original/    ← Клонированный репозиторий RecursiveMAS (git submodule)
└── custom_impl/     ← Собственная реализация
    ├── train_inner.py              ← Inner-Loop Training (Sequential-Math)
    ├── train_outer.py              ← Outer-Loop Training (Sequential-Math)
    ├── gsm8k_utils.py              ← Утилиты для оценки на GSM8K
    ├── evaluate_single_model.py    ← Baseline: одна модель
    ├── evaluate_text_mas.py        ← Текстовый MAS (Planner -> Critic -> Solver)
    └── evaluate_recursivemas.py    ← RecursiveMAS (латентная коммуникация)
```

---

## Обучение (Training Pipeline)

> **Датасет:** `RecursiveMAS/Sequential-Math` (HuggingFace, 1904 примера) — тот же, что в оригинальной статье.

### Запуск с видимыми логами

```bash
cd /home/hermes/workspace/projects/recursivemas
source .venv

# Inner-Loop Training:
python3 -u custom_impl/train_inner.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --mas_role planner \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir ~/workspace/tmp/recursivemas/checkpoints/inner_planner \
    --max_steps 20000 \
    --batch_size 2 \
    --device cuda:2 \
    2>&1 | tee ~/workspace/tmp/recursivemas/train_inner_planner.log

# Outer-Loop Training:
python3 -u custom_impl/train_outer.py \
    --agent1_model Qwen/Qwen3-1.7B \
    --agent2_model Qwen/Qwen3-1.7B \
    --agent3_model /home/hermes/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct \
    --agent1_inner ~/workspace/tmp/recursivemas/checkpoints/inner_planner/ \
    --agent2_inner ~/workspace/tmp/recursivemas/checkpoints/inner_refiner/ \
    --agent3_inner ~/workspace/tmp/recursivemas/checkpoints/inner_solver/ \
    --save_dir ~/workspace/tmp/recursivemas/checkpoints/outer/ \
    --num_recursive_rounds 3 \
    --max_length 4096 \
    --max_latent_tokens 80 \
    --outer_adapter_type outer_ln_res_adapter \
    --supervise_final_only 1 \
    --batch_size 1 \
    --grad_accum_steps 2 \
    --max_steps 20000 \
    --outer_lr 5e-4 \
    --device cuda:2 \
    --dtype bfloat16 \
    --outer_dtype bfloat16 \
    2>&1 | tee ~/workspace/tmp/recursivemas/train_outer.log
```

> **Важно:** `-u` принудительно отключает буферизацию Python. Вывод через `tee` — логи видны в реальном времени и сохраняются в файл.
> **Стиль:** Sequential (Planner -> Refiner/Critic -> Solver)
> **GPU:** cuda:2 (RTX 4060 Ti 16GB)
> **Промпты:** оригинальные из `mas_prompt.py`

### Шаг 1: Inner-Loop Training

Обучаем inner adapter для каждой роли отдельно:

```bash
# Planner (Qwen3-1.7B)
python custom_impl/train_inner.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --mas_role planner \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir ~/workspace/tmp/recursivemas/checkpoints/inner_planner \
    --max_steps 20000

# Refiner/Critic (Qwen3-1.7B, замена Llama-3.2-1B)
python custom_impl/train_inner.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --mas_role refiner \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir ~/workspace/tmp/recursivemas/checkpoints/inner_refiner \
    --max_steps 20000

# Solver (Qwen2.5-Math-1.5B-Instruct)
python custom_impl/train_inner.py \
    --model_name_or_path /home/hermes/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct \
    --mas_role solver \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir ~/workspace/tmp/recursivemas/checkpoints/inner_solver \
    --max_steps 20000
```

**Дополнительные опции:**

- `--dataset_name openai/gsm8k` — обучение на GSM8K (для отладки)
- `--load_dir ckpts/` — продолжение обучения с чекпоинта
- `--grad_accum_steps 4` — накопление градиентов
- `--enable_thinking 1` — thinking mode для токенизатора
- `--solver_pre_question 1` — вопрос перед планом в solver prompt

**Параметры (как в оригинале):**

- `--adapter_lr 5e-4` — learning rate
- `--lr_scheduler_type cosine` — scheduler
- `--warmup_steps 10` — warmup
- `--adapter_cos_weight 1.0` — вес cosine loss
- `--adapter_mse_weight 0.0` — вес MSE loss (0 по умолчанию)
- `--dtype bfloat16` — precision

### Шаг 2: Outer-Loop Training

Соединяем агенты и обучаем outer adapters:

```bash
python custom_impl/train_outer.py \
    --agent1_model Qwen/Qwen3-1.7B \
    --agent2_model Qwen/Qwen3-1.7B \
    --agent3_model /home/hermes/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct \
    --agent1_inner ~/workspace/tmp/recursivemas/checkpoints/inner_planner \
    --agent2_inner ~/workspace/tmp/recursivemas/checkpoints/inner_refiner \
    --agent3_inner ~/workspace/tmp/recursivemas/checkpoints/inner_solver \
    --save_dir ~/workspace/tmp/recursivemas/checkpoints/outer \
    --max_steps 20000 \
    --outer_lr 5e-4 \
    --num_recursive_rounds 3
```

**Параметры (как в оригинале):**

- `--max_length 4096` — максимальная длина последовательности
- `--max_latent_tokens 80` — максимальное количество латентных токенов
- `--num_recursive_rounds 3` — количество рекурсивных итераций
- `--supervise_final_only 1` — оптимизировать только последний round
- `--outer_adapter_type outer_ln_res_adapter` — тип outer adapter (LayerNorm + residual)

### Модели (Sequential Light)

| Роль | Модель | Hidden size | Примечание |
|------|--------|-------------|------------|
| Planner | Qwen/Qwen3-1.7B | 2048 | |
| Refiner/Critic | Qwen/Qwen3-1.7B | 2048 | Замена Llama-3.2-1B (gated repo) |
| Solver | Qwen2.5-Math-1.5B-Instruct | 1536 | |

---

## Оценка (Evaluation)

> **Датасет:** GSM8K test split (1319 примеров)
> **Метрика:** Exact Match (парсинг `#### X`)

```bash
# Baseline: одна модель
python custom_impl/evaluate_single_model.py --num_examples 100

# Текстовый MAS (Planner -> Critic -> Solver через текст)
python custom_impl/evaluate_text_mas.py --num_examples 100

# RecursiveMAS (с обученными адаптерами)
python custom_impl/evaluate_recursivemas.py \
    --num_examples 1319 \
    --outer_checkpoint ~/workspace/tmp/recursivemas/checkpoints/outer/checkpoint-20000/ \
    --inner1_checkpoint ~/workspace/tmp/recursivemas/checkpoints/inner_planner/ \
    --inner2_checkpoint ~/workspace/tmp/recursivemas/checkpoints/inner_refiner/ \
    --inner3_checkpoint ~/workspace/tmp/recursivemas/checkpoints/inner_solver/ \
    --num_latent_tokens 32 \
    --num_recursive_rounds 3
```

### Результаты

**Реальный результат (2025-07-09):**

- `evaluate_recursivemas.py` с обученными outer adapters (checkpoint-20000): **81.8% (1079/1319)** на GSM8K test (полный датасет)
- Среднее время: 30.44s/пример
- Pipeline: Planner -> inner_1 -> outer_12 -> Refiner -> inner_2 -> outer_23 -> Solver

---

## Соответствие оригиналу

Наша реализация максимально приближена к оригинальной:

| Компонент | Статус | Описание |
| --- | --- | --- |
| Inner adapter | Совместим | `ln_res_adapter` (LayerNorm -> MLP -> residual -> LayerNorm) |
| Outer adapter | Совместим | `outer_ln_res_adapter` (LayerNorm + hidden_dim=out_dim*2 + residual) |
| Inner dataset | Совместим | `RecursiveMAS/Sequential-Math` (HuggingFace, 1904 примера) |
| Outer dataset | Совместим | `RecursiveMAS/Sequential-Math` (HuggingFace, 1904 примера) |
| Inner loss | Совместим | Cosine similarity (weight=1.0) + MSE (weight=0.0) |
| Outer loss | Совместим | CE loss на solver output, `supervise_final_only=1` |
| Prompts | Совместим | Оригинальные из `mas_prompt.py` |
| Recursive rounds | Совместим | 3 round'а (planner -> refiner -> solver -> feedback) |
| max_length | Совместим | 4096 (outer), 2048 (inner) |
| Optimizer | Совместим | AdamW(betas=(0.9, 0.95)) |
| Scheduler | Совместим | Cosine with warmup, 10 steps |

### Отличия от оригинала

- **Refiner модель:** `Qwen/Qwen3-1.7B` вместо `meta-llama/Llama-3.2-1B-Instruct` (gated repo)
- **Без accelerate:** используем прямой PyTorch (single GPU)
- **Претокенизация:** весь датасет токенизируется upfront в inner loop
- **Дополнительные флаги:** `--grad_accum_steps`, `--load_dir`, `--enable_thinking`, `--solver_pre_question`

---

## Ключевые концепции

### 4 стиля коллаборации

| Стиль | Агенты | Описание |
|-------|--------|----------|
| **Sequential** | Planner -> Critic -> Solver | Последовательная цепочка: планирование -> критика -> решение |
| **Mixture** | Math + Code + Science experts -> Summarizer | Смешивание экспертов с агрегацией |
| **Distillation** | Expert (9B) -> Learner (4B) | Передача знаний от большого к малому |
| **Deliberation** | Reflector + Tool-Caller | Итеративное обсуждение с внешними инструментами |

### Почему латентное пространство?

1. **Эффективность** — `d_h << |V|`, поэтому `m*d_h^2` намного дешевле `m*|V|*d_h`
2. **Стабильность градиентов** — остаточное соединение в RecursiveLink сохраняет градиенты около 1, в то время как текстовая передача приводит к затуханию
