# RecursiveMAS — Изучение и эксперименты

> **Recursive Multi-Agent Systems** — фреймворк для масштабирования коллаборации агентов через рекурсию в латентном пространстве.

**Статья:** [arXiv:2604.25917](https://arxiv.org/abs/2604.25917)  
**Репозиторий:** [RecursiveMAS/RecursiveMAS](https://github.com/RecursiveMAS/RecursiveMAS)  
**Модели:** [HuggingFace/RecursiveMAS](https://huggingface.co/RecursiveMAS/models)  
**Демо:** [Playground](https://vishalmysore.github.io/recursiveMASDemo/)

---

## Что такое RecursiveMAS?

Обычные мульти-агентные системы (MAS) передают информацию между агентами через **текст**: агент A генерирует текст → агент B читает текст → и так далее. Это дорого (много токенов), медленно (каждый агент декодирует текст) и плохо обучается (градиенты затухают через текстовые границы).

**RecursiveMAS** заменяет текстовую коммуникацию на **латентную**: агенты обмениваются векторными представлениями (embeddings) напрямую, минуя декодирование в текст. Ключевые компоненты:

### 🔗 RecursiveLink
- **Inner Link** — внутри каждого агента: преобразует last-layer hidden state обратно в input embedding space, позволяя агенту генерировать «латентные мысли» без текстового декодирования
- **Outer Link** — между агентами: проецирует латентные представления одного агента в пространство другого, даже если модели разной архитектуры и размера

### 🔄 Inner-Outer Loop Training
1. **Inner Loop** — каждый агент обучается генерировать латентные мысли, близкие к embedding'ам ground-truth ответа (косинусная регрессия)
2. **Outer Loop** — вся система обучается как единое целое: градиенты проходят через все рекурсивные итерации, оптимизируя только Outer Link модули

### 📊 Результаты
- **+8.3%** средняя точность vs лучшие базовые методы
- **1.2×–2.4×** ускорение инференса
- **34.6%–75.6%** сокращение использования токенов

---

## Структура проекта

```
recursivemas/
├── PLAN.md          ← Этапный план изучения
├── README.md        ← Этот файл
├── AGENTS.md        ← Контекст для агента
├── INSTALL.md       ← Инструкция по развёртыванию
├── theory/          ← Теоретические заметки и конспекты
├── experiments/     ← Воспроизведение результатов из статьи
│   └── original/    ← Клонированный репозиторий RecursiveMAS
├── custom_impl/     ← Собственная реализация
│   ├── train_inner.py              ← Inner-Loop Training (Sequential-Math)
│   ├── train_outer.py              ← Outer-Loop Training (Sequential-Math)
│   ├── gsm8k_utils.py              ← Утилиты для оценки на GSM8K
│   ├── evaluate_single_model.py    ← Baseline: одна модель
│   ├── evaluate_text_mas.py        ← Текстовый MAS (Planner → Critic → Solver)
│   ├── evaluate_recursivemas.py    ← RecursiveMAS (латентная коммуникация, обновлён для train_outer.py)
│   │
│   │ (устаревшие — см. *_ARCHIVE.md)
│   ├── inner_link_training.py      ← ⚠️ УСТАРЕЛ
│   ├── outer_link_training_v2.py   ← ⚠️ УСТАРЕЛ
│   ├── pregenerate_teacher_data.py ← ⚠️ УСТАРЕЛ
│   ├── test_pipeline.py            ← Базовые тесты
│   ├── test_slot_injection.py      ← Тест пайплайна
│   └── checkpoints/                ← Чекпоинты обученных адаптеров
├── models/          ← Локальные модели (Qwen2.5-Math-1.5B-Instruct)
└── notebooks/       ← Jupyter-ноутбуки для исследования
```

---

## Обучение (Training Pipeline)

> **Датасет:** `RecursiveMAS/Sequential-Math` (HuggingFace, 1904 примера)
> **Стиль:** Sequential (Planner → Refiner/Critic → Solver)
> **GPU:** cuda:2 (RTX 4060 Ti 16GB)
> **Промпты:** оригинальные из `mas_prompt.py` (обновлено 2025-07-09)
>
> **🔄 Статус:** Требуется переобучение с оригинальными промптами.

### Шаг 1: Inner-Loop Training 🔄 ТРЕБУЕТСЯ ПЕРЕОБУЧЕНИЕ

Обучаем inner adapter для каждой роли отдельно:

```bash
# Planner (Qwen3-1.7B)
python custom_impl/train_inner.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --mas_role planner \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir custom_impl/checkpoints/inner_planner \
    --max_steps 20000

# Refiner/Critic (Qwen3-1.7B, замена Llama-3.2-1B)
python custom_impl/train_inner.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --mas_role refiner \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir custom_impl/checkpoints/inner_refiner \
    --max_steps 20000

# Solver (Qwen2.5-Math-1.5B-Instruct)
python custom_impl/train_inner.py \
    --model_name_or_path ./models/Qwen2.5-Math-1.5B-Instruct \
    --mas_role solver \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir custom_impl/checkpoints/inner_solver \
    --max_steps 20000
```

**Результаты предыдущего обучения (на модифицированных промптах):**

| Роль | Final Loss | Время |
|------|-----------|-------|
| Planner | 0.254 (cos) | ~30 мин |
| Refiner | 0.245 (cos) | ~46 мин |
| Solver | 0.171 (cos) | ~79 мин |

**Дополнительные опции:**
- `--dataset_name openai/gsm8k` — обучение на GSM8K (для отладки)
- `--load_dir ckpts/` — продолжение обучения с чекпоинта
- `--grad_accum_steps 4` — накопление градиентов
- `--enable_thinking 1` — thinking mode для токенизатора
- `--solver_pre_question 1` — вопрос перед планом в solver prompt

**Параметры:**
- `--adapter_lr 5e-4` — learning rate (как в оригинале)
- `--lr_scheduler_type cosine` — scheduler
- `--warmup_steps 10` — warmup
- `--adapter_cos_weight 1.0` — вес cosine loss
- `--adapter_mse_weight 0.0` — вес MSE loss (0 по умолчанию)
- `--dtype bfloat16` — precision

### Шаг 2: Outer-Loop Training 🔄 ТРЕБУЕТСЯ ПЕРЕОБУЧЕНИЕ

Соединяем агенты и обучаем outer adapters:

```bash
python custom_impl/train_outer.py \
    --agent1_model Qwen/Qwen3-1.7B \
    --agent2_model Qwen/Qwen3-1.7B \
    --agent3_model ./models/Qwen2.5-Math-1.5B-Instruct \
    --agent1_inner custom_impl/checkpoints/inner_planner \
    --agent2_inner custom_impl/checkpoints/inner_refiner \
    --agent3_inner custom_impl/checkpoints/inner_solver \
    --save_dir custom_impl/checkpoints/outer \
    --max_steps 20000 \
    --outer_lr 5e-4 \
    --num_recursive_rounds 3
```

**Результаты предыдущего обучения:**

| Компонент | Final Loss | Чекпоинт |
|-----------|-----------|----------|
| Outer adapters (3) | ~0.5 (CE) | `custom_impl/checkpoints/outer/` |

**Оценка на GSM8K test (1319 примеров):**

| Метрика | Значение |
|---------|----------|
| **Accuracy** | **78.2% (1031/1319)** |
| Среднее время | 8.78s/пример |
| Pipeline | Planner → inner_1 → outer_12 → Refiner → inner_2 → outer_23 → Solver |

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

# Текстовый MAS (Planner → Critic → Solver через текст)
python custom_impl/evaluate_text_mas.py --num_examples 100

# RecursiveMAS (с обученными адаптерами)
python custom_impl/evaluate_recursivemas.py \
    --num_examples 100 \
    --checkpoint custom_impl/checkpoints/outer/outer_adapters.pt
```

### Ожидаемая иерархия результатов

```
evaluate_text_mas.py       →  ~65-75%  (текстовый MAS, лучше baseline)
evaluate_single_model.py   →  ~50-60%  (Qwen2.5-Math-1.5B baseline)
evaluate_single_model.py   →  ~30-40%  (Qwen3-1.7B baseline)
evaluate_recursivemas.py   →  ~78%     (обученные адаптеры, РЕАЛЬНЫЙ РЕЗУЛЬТАТ)
```

**Реальный результат (2025-07-09):**
- `evaluate_recursivemas.py` с обученными outer adapters: **78.2% (1031/1319)** на GSM8K test (полный датасет)
- ⚠️ Результат получен на чекпоинтах, обученных с модифицированными промптами. После переобучения с оригинальными промптами результат может измениться.

---

## Ключевые концепции

### 4 стиля коллаборации
| Стиль | Агенты | Описание |
|-------|--------|----------|
| **Sequential** | Planner → Critic → Solver | Последовательная цепочка: планирование → критика → решение |
| **Mixture** | Math + Code + Science experts → Summarizer | Смешивание экспертов с агрегацией |
| **Distillation** | Expert (9B) → Learner (4B) | Передача знаний от большого к малому |
| **Deliberation** | Reflector + Tool-Caller | Итеративное обсуждение с внешними инструментами |

### Почему латентное пространство?
1. **Эффективность** — `d_h << |V|`, поэтому `m·d_h²` намного дешевле `m·|V|·d_h`
2. **Стабильность градиентов** — остаточное соединение в RecursiveLink сохраняет градиенты ≈1, в то время как текстовая передача приводит к затуханию ≈ε

### Модели (примеры)
- **Sequential Light**: Qwen3-1.7B + Llama3.2-1B + Qwen2.5-Math-1.5B (~4.2B суммарно)
- **Sequential Scaled**: Gemma3-4B + Llama3.2-3B + Qwen3.5-4B (~11B суммарно)
