# TODO.md — Обучение и тестирование RecursiveMAS

## Статус

> **🔄 ОБНОВЛЕНО (2025-07-08):** Inner-Loop Training завершён.
> Обучение: `RecursiveMAS/Sequential-Math` (HuggingFace)
> Оценка: GSM8K test (без изменений)

---

## Обучение (Training)

### Скрипты

| Скрипт | Статус | Датасет | Описание |
|--------|--------|---------|----------|
| `train_inner.py` | ✅ Актуален | Sequential-Math / GSM8K | Inner-Loop Training (как в оригинале) |
| `train_outer.py` | ✅ Актуален | Sequential-Math | Outer-Loop Training (как в оригинале) |
| `inner_link_training.py` | ⚠️ УСТАРЕЛ | Синтетические | Заменён на `train_inner.py` |
| `outer_link_training_v2.py` | ⚠️ УСТАРЕЛ | GSM8K | Заменён на `train_outer.py` |
| `pregenerate_teacher_data.py` | ⚠️ УСТАРЕЛ | GSM8K | Артефакт, см. `*_ARCHIVE.md` |

### Inner-Loop Training — ✅ ЗАВЕРШЁНО

Все три роли обучены на `RecursiveMAS/Sequential-Math` (1904 примера).

| Роль | Модель | Final Loss | Время | Чекпоинт |
|------|--------|-----------|-------|----------|
| **Planner** | Qwen3-1.7B | **0.254** (cos) | ~30 мин | `custom_impl/checkpoints/inner_planner/` |
| **Refiner** | Qwen3-1.7B | **0.245** (cos) | ~46 мин | `custom_impl/checkpoints/inner_refiner/` |
| **Solver** | Qwen2.5-Math-1.5B | **0.171** (cos) | ~79 мин | `custom_impl/checkpoints/inner_solver/` |

**Параметры:** 20000 steps, batch_size=2, LR=5e-4, cosine scheduler, warmup=10, bfloat16

**Примечание:** Refiner обучен на Qwen3-1.7B вместо Llama-3.2-1B (gated repo). Hidden size одинаковый (2048).

### Outer-Loop Training — ✅ ЗАВЕРШЕНО (2025-07-08)

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
    --num_recursive_rounds 3 \
    --device cuda:2
```

**Результаты обучения:**
- Final loss: ~0.5 (CE loss)
- Чекпоинты: `custom_impl/checkpoints/outer/`
- Outer adapters: outer_12 (Planner→Refiner), outer_23 (Refiner→Solver), outer_31 (Solver→Planner)

**Оценка на GSM8K test (20 примеров):**
- **70% accuracy (14/20)** — с обученными outer adapters
- Среднее время на пример: ~9.3s
- Pipeline: Planner → inner_1 → outer_12 → Refiner → inner_2 → outer_23 → Solver

---

## Тестирование (Evaluation) — без изменений

### Скрипты

| Скрипт | Описание |
|--------|----------|
| `evaluate_single_model.py` | Baseline: одна модель |
| `evaluate_text_mas.py` | Текстовый MAS (Planner → Critic → Solver) |
| `evaluate_recursivemas.py` | RecursiveMAS (латентная коммуникация) |

### Общие утилиты (`gsm8k_utils.py`)

| Функция | Назначение |
|---------|-----------|
| `load_gsm8k_dataset(num_examples, split)` | Загрузка GSM8K из HuggingFace |
| `parse_answer(text)` | Парсинг `#### X` из ответа |
| `compute_accuracy(predictions, answers, questions)` | Вычисление accuracy + детальные примеры |
| `print_results(results, title)` | Красивый вывод в консоль |
| `save_results(results, filepath)` | Сохранение в JSON |
| `load_model(model_path, device)` | Загрузка модели + токенизатора |
| `generate_answer(model, tokenizer, prompt, device, **kwargs)` | Генерация ответа |
| `format_gsm8k_prompt(question, system_msg)` | Форматирование prompt для GSM8K |
| `timeit(func, *args, **kwargs)` | Замер времени выполнения |

### Модели

| Роль | Модель | Hidden size | Путь |
|------|--------|-------------|------|
| Planner | Qwen3-1.7B | 2048 | `Qwen/Qwen3-1.7B` (HF cache) |
| Critic | Qwen3-1.7B | 2048 | `Qwen/Qwen3-1.7B` (та же модель) |
| Solver | Qwen2.5-Math-1.5B-Instruct | 1536 | `./models/Qwen2.5-Math-1.5B-Instruct` |

**GPU:** cuda:2 (RTX 4060 Ti 16GB)

### Ожидаемая иерархия результатов

```
evaluate_text_mas.py       →  ~65-75%  (текстовый MAS, лучше baseline)
evaluate_single_model.py   →  ~50-60%  (Qwen2.5-Math-1.5B baseline)
evaluate_single_model.py   →  ~30-40%  (Qwen3-1.7B baseline)
evaluate_recursivemas.py   →  ~70%     (обученные адаптеры, РЕАЛЬНЫЙ РЕЗУЛЬТАТ)
```

**Реальный результат (2025-07-08):**
- `evaluate_recursivemas.py` с обученными outer adapters: **70% (14/20)** на GSM8K test
- Pipeline: Planner → inner_1 → outer_12 → Refiner → inner_2 → outer_23 → Solver
- Среднее время на пример: ~9.3s
