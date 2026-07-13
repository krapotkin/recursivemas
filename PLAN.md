# PLAN.md — План работ по проекту RecursiveMAS

## Текущий статус (2025-07-09)

### Завершено

- Изучение оригинальной статьи и репозитория
- Клонирование оригинального репозитория в `experiments/original/`
- Реализация Inner-Loop Training (`train_inner.py`) — совместима с оригиналом
- Реализация Outer-Loop Training (`train_outer.py`) — полностью переписана, совместима с оригиналом
- Обучение Inner-Loop (3 роли, 20000 steps каждая)
- Обучение Outer-Loop (20000 steps)
- Оценка на GSM8K test: **81.8% accuracy (1079/1319)**
- Документация архивных файлов (`pregenerate_teacher_data.py`)

### Что изменилось

**До:** Использовали `openai/gsm8k` для обучения + `pregenerate_teacher_data.py` для генерации планов.
**После:** Используем `RecursiveMAS/Sequential-Math` (как в оригинале) — датасет уже содержит teacher-generated планы.

### Архитектура (совместима с оригиналом)

| Компонент | Статус |
| --- | --- |
| Inner adapter (`ln_res_adapter`) | Совместим |
| Outer adapter (`outer_ln_res_adapter`) | Совместим |
| Dataset (`RecursiveMAS/Sequential-Math`) | Совместим |
| Inner loss (cosine + MSE) | Совместим |
| Outer loss (CE, supervise_final_only) | Совместим |
| Prompts (из `mas_prompt.py`) | Совместим |
| Recursive rounds (3) | Совместим |
| max_length (4096) | Совместим |
| Optimizer (AdamW) | Совместим |
| Scheduler (cosine + warmup) | Совместим |

### Отличия от оригинала

- **Refiner модель:** `Qwen/Qwen3-1.7B` вместо `meta-llama/Llama-3.2-1B-Instruct` (gated repo)
- **Без accelerate:** используем прямой PyTorch (single GPU)
- **Претокенизация:** весь датасет токенизируется upfront в inner loop

---

## Дальнейшие шаги

### Приоритет 1: Переобучение с оригинальными датасетами

Inner и Outer loop уже используют `RecursiveMAS/Sequential-Math`, но чекпоинты обучены на старых данных. Нужно:

1. Переобучить Inner-Loop (3 роли) на `RecursiveMAS/Sequential-Math`
2. Переобучить Outer-Loop на `RecursiveMAS/Sequential-Math`
3. Оценить на GSM8K test

Ожидаемый результат: улучшение accuracy за счёт более качественных teacher-generated планов.

### Приоритет 2: Baseline сравнения

- `evaluate_single_model.py` — Qwen2.5-Math-1.5B без адаптеров (полный GSM8K test)
- `evaluate_text_mas.py` — текстовый MAS (полный GSM8K test)
- Сравнить с RecursiveMAS (81.8%)

### Приоритет 3: Ablation study

- Без Outer Link 1 (Planner -> Refiner)
- Без Outer Link 2 (Refiner -> Solver)
- Разная глубина рекурсии (r=1 vs r=3)

### Приоритет 4: Оценка на других бенчмарках (из оригинала)

- MATH-500 (`math500`)
- GPQA (`gpqa`)
- MedQA (`medqa`)
- AIME 2025/2026 (`aime25`, `aime26`)

---

## Архивные файлы

| Файл | Статус | Описание |
| --- | --- | --- |
| `inner_link_training.py` | УСТАРЕЛ | Заменён на `train_inner.py` |
| `outer_link_training_v2.py` | УСТАРЕЛ | Заменён на `train_outer.py` |
| `pregenerate_teacher_data.py` | УСТАРЕЛ | Генерация планов через GSM8K. Не используется, так как оригинал использует `RecursiveMAS/Sequential-Math` с teacher-generated планами. См. `pregenerate_teacher_data_ARCHIVE.md`. |
