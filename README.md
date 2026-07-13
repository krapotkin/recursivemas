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
├── README.md        <-- Этот файл
├── AGENTS.md        <-- Контекст для агента
├── INSTALL.md       <-- Инструкция по развёртыванию
├── DATA.md          <-- Описание данных и результатов
├── theory/          <-- Теоретические заметки и конспекты
├── experiments/     <-- Воспроизведение результатов из статьи
│   └── original/    <-- Клонированный репозиторий RecursiveMAS (git submodule)
├── custom_impl/     <-- Собственная реализация
│   ├── train_inner.py              <-- Inner-Loop Training (Sequential-Math)
│   ├── train_outer.py              <-- Outer-Loop Training (Sequential-Math)
│   ├── gsm8k_utils.py              <-- Утилиты для GSM8K (загрузка, парсинг)
│   ├── evaluate_single_model.py    <-- Baseline: одна модель на GSM8K
│   ├── evaluate_text_mas.py        <-- Текстовый MAS (Planner -> Refiner -> Solver)
│   └── evaluate_recursivemas.py    <-- RecursiveMAS (латентная коммуникация)
└── tests/           <-- Unit-тесты
```

---

## Запуск процедур

Все процедуры запускаются с флагом `-u` (unbuffered) и выводом через `tee` — логи видны в реальном времени и сохраняются в файл.

**Шаблон запуска:**

```bash
cd /home/hermes/workspace/projects/recursivemas
source .venv

python3 -u custom_impl/<скрипт>.py [аргументы] \
    2>&1 | tee ~/workspace/tmp/recursivemas/<имя>.log
```

**GPU:** cuda:2 (RTX 4060 Ti 16GB)
**Датасет для обучения:** `RecursiveMAS/Sequential-Math` (1904 примера)
**Датасет для оценки:** `openai/gsm8k` test split (1319 примеров)
**Чекпоинты:** `~/workspace/tmp/recursivemas/checkpoints/`
**Результаты:** `~/workspace/data/recursivemas/results/`
**Модели:** `~/workspace/models/models_llm/`

---

## 1. Оценка одиночной модели (evaluate_single_model.py)

**Что делает:** Загружает одну модель без адаптеров и оценивает точность на GSM8K test. Базовый baseline для сравнения с MAS-системами.

```bash
# Qwen2.5-Math-1.5B-Instruct (математическая модель):
python3 -u custom_impl/evaluate_single_model.py \
    --model /home/hermes/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct \
    --num_examples 1319 \
    --device cuda:2 \
    --output ~/workspace/data/recursivemas/results/eval_single_qwen25_math_15b.json \
    2>&1 | tee ~/workspace/tmp/recursivemas/eval_single_qwen25_math_15b.log

# Qwen3-1.7B (общая модель):
python3 -u custom_impl/evaluate_single_model.py \
    --model /home/hermes/workspace/models/models_llm/Qwen3-1.7B \
    --num_examples 1319 \
    --device cuda:2 \
    --output ~/workspace/data/recursivemas/results/eval_single_qwen3_17b.json \
    2>&1 | tee ~/workspace/tmp/recursivemas/eval_single_qwen3_17b.log
```

**Аргументы:**
- `--model` — путь к модели (локальный) или имя на HuggingFace
- `--num_examples` — количество примеров (1319 = полный датасет)
- `--device` — GPU устройство (cuda:2)
- `--output` — путь к JSON файлу с результатами

**Время выполнения:** ~2.5-3 часа (1319 примеров)

---

## 2. Оценка текстового MAS (evaluate_text_mas.py)

**Что делает:** Запускает три модели последовательно через текстовую коммуникацию:
Planner (Qwen3-1.7B) -> Refiner (Qwen3-1.7B) -> Solver (Qwen2.5-Math-1.5B).
Каждый агент генерирует текст, который передаётся следующему.

```bash
python3 -u custom_impl/evaluate_text_mas.py \
    --num_examples 1319 \
    --device cuda:2 \
    --output ~/workspace/data/recursivemas/results/eval_text_mas.json \
    2>&1 | tee ~/workspace/tmp/recursivemas/eval_text_mas.log
```

**Аргументы:**
- `--num_examples` — количество примеров (1319 = полный датасет)
- `--device` — GPU устройство (cuda:2)
- `--output` — путь к JSON файлу с результатами

**Время выполнения:** ~8-10 часов (1319 примеров, 3 модели последовательно)

---

## 3. Обучение Inner Link (train_inner.py)

**Что делает:** Обучает inner adapter для одной роли. Adapter преобразует hidden state модели обратно в embedding space, позволяя генерировать "латентные мысли".

**Параметры по умолчанию (как в оригинале):**
- Adapter: `ln_res_adapter` (LayerNorm -> MLP -> residual -> LayerNorm)
- Loss: Cosine similarity (weight=1.0) + MSE (weight=0.1)
- Optimizer: AdamW (betas=(0.9, 0.95))
- Scheduler: Cosine с warmup (10 steps)
- Learning rate: 5e-4

```bash
# Planner (Qwen3-1.7B):
python3 -u custom_impl/train_inner.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --mas_role planner \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir ~/workspace/tmp/recursivemas/checkpoints/inner_planner \
    --max_steps 20000 \
    --batch_size 2 \
    --device cuda:2 \
    2>&1 | tee ~/workspace/tmp/recursivemas/train_inner_planner.log

# Refiner (Qwen3-1.7B):
python3 -u custom_impl/train_inner.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --mas_role refiner \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir ~/workspace/tmp/recursivemas/checkpoints/inner_refiner \
    --max_steps 20000 \
    --batch_size 2 \
    --device cuda:2 \
    2>&1 | tee ~/workspace/tmp/recursivemas/train_inner_refiner.log

# Solver (Qwen2.5-Math-1.5B-Instruct):
python3 -u custom_impl/train_inner.py \
    --model_name_or_path /home/hermes/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct \
    --mas_role solver \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir ~/workspace/tmp/recursivemas/checkpoints/inner_solver \
    --max_steps 20000 \
    --batch_size 2 \
    --device cuda:2 \
    2>&1 | tee ~/workspace/tmp/recursivemas/train_inner_solver.log
```

**Дополнительные опции:**
- `--load_dir` — продолжение обучения с чекпоинта
- `--grad_accum_steps` — накопление градиентов
- `--enable_thinking 1` — thinking mode для токенизатора
- `--solver_pre_question 1` — вопрос перед планом в solver prompt
- `--adapter_mse_weight 0.1` — вес MSE loss (default: 0.1)

**Время обучения:** ~4-6 часов на роль (20000 steps)

---

## 4. Обучение Outer Link (train_outer.py)

**Что делает:** Соединяет три агента с обученными inner adapters и обучает outer adapters для латентной коммуникации между ними. Градиенты проходят через все рекурсивные итерации.

**Параметры по умолчанию (как в оригинале):**
- Outer adapter: `outer_ln_res_adapter` (LayerNorm + hidden_dim=out_dim*2 + residual)
- Loss: CE на solver output, `supervise_final_only=1`
- Recursive rounds: 3
- Optimizer: AdamW (betas=(0.9, 0.95))
- Learning rate: 5e-4

```bash
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

**Время обучения:** ~12-16 часов (20000 steps)

---

## 5. Оценка RecursiveMAS (evaluate_recursivemas.py)

**Что делает:** Запускает полную систему с обученными inner + outer adapters. Агенты обмениваются латентными представлениями напрямую без текстового декодирования.

**Pipeline:** Planner -> inner_1 -> outer_12 -> Refiner -> inner_2 -> outer_23 -> Solver

```bash
python3 -u custom_impl/evaluate_recursivemas.py \
    --agent1_model Qwen/Qwen3-1.7B \
    --agent2_model Qwen/Qwen3-1.7B \
    --agent3_model /home/hermes/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct \
    --inner1_checkpoint ~/workspace/tmp/recursivemas/checkpoints/inner_planner/ \
    --inner2_checkpoint ~/workspace/tmp/recursivemas/checkpoints/inner_refiner/ \
    --inner3_checkpoint ~/workspace/tmp/recursivemas/checkpoints/inner_solver/ \
    --outer_checkpoint ~/workspace/tmp/recursivemas/checkpoints/outer/checkpoint-20000/ \
    --num_examples 1319 \
    --num_latent_tokens 32 \
    --num_recursive_rounds 3 \
    --device cuda:2 \
    --output ~/workspace/data/recursivemas/results/eval_recursivemas.json \
    2>&1 | tee ~/workspace/tmp/recursivemas/eval_recursivemas.log
```

**Аргументы:**
- `--agent[1-3]_model` — модели для Planner, Refiner, Solver
- `--inner[1-3]_checkpoint` — чекпоинты inner adapters
- `--outer_checkpoint` — чекпоинт outer adapters
- `--num_latent_tokens` — количество латентных токенов (32)
- `--num_recursive_rounds` — рекурсивные итерации (3)
- `--output` — путь к JSON файлу с результатами

**Время выполнения:** ~12-14 часов (1319 примеров)

---

## Модели (Sequential Light)

| Роль | Модель | Hidden size | Путь |
|------|--------|-------------|------|
| Planner | Qwen/Qwen3-1.7B | 2048 | `~/workspace/models/models_llm/Qwen3-1.7B/` |
| Refiner | Qwen/Qwen3-1.7B | 2048 | `~/workspace/models/models_llm/Qwen3-1.7B/` |
| Solver | Qwen2.5-Math-1.5B-Instruct | 1536 | `~/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct/` |

---

## Результаты оценки

| Метод | Точность | Время/пример | Статус |
|-------|----------|--------------|--------|
| Qwen2.5-Math-1.5B (single) | 83.2% (1098/1319) | 7.51s | выполнено |
| Qwen3-1.7B (single) | в процессе | -- | в процессе |
| Text MAS (3 модели) | -- | -- | ожидает |
| RecursiveMAS (latency) | -- | -- | ожидает |

---

## Соответствие оригиналу

Наша реализация максимально приближена к оригинальной:

| Компонент | Статус | Описание |
| --- | --- | --- |
| Inner adapter | Совместим | `ln_res_adapter` (LayerNorm -> MLP -> residual -> LayerNorm) |
| Outer adapter | Совместим | `outer_ln_res_adapter` (LayerNorm + hidden_dim=out_dim*2 + residual) |
| Inner dataset | Совместим | `RecursiveMAS/Sequential-Math` (HuggingFace, 1904 примера) |
| Outer dataset | Совместим | `RecursiveMAS/Sequential-Math` (HuggingFace, 1904 примера) |
| Inner loss | Совместим | Cosine similarity (weight=1.0) + MSE (weight=0.1) |
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
