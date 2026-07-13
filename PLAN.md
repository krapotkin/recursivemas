# PLAN.md — План работ по проекту RecursiveMAS

## Текущий статус (2025-07-13, ветка clean)

### Выполнено

- Изучение оригинальной статьи и репозитория
- Клонирование оригинального репозитория в `experiments/original/`
- Реализация Inner-Loop Training (`train_inner.py`) — совместима с оригиналом
- Реализация Outer-Loop Training (`train_outer.py`) — полностью переписана, совместима с оригиналом
- Обучение Inner-Loop (3 роли, 20000 steps каждая)
- Обучение Outer-Loop (20000 steps)
- Оценка на GSM8K test: **81.8% accuracy (1079/1319)** — предыдущий результат

### Выполнена очистка проекта (ветка clean)

- Удалены устаревшие файлы (10 файлов)
- Чекпоинты перемещены в `~/workspace/tmp/recursivemas/old_checkpoints/`
- Удалены пустые папки `models/`, `notebooks/`
- Исправлены пути к моделям (теперь `~/workspace/models/models_llm/`)
- Убраны эмодзи из кода
- Исправлены импорты и мелкие проблемы (float16 -> bfloat16, dataset name)
- Создан `.gitattributes`, обновлён `.gitignore`
- Написаны unit-тесты (40 тестов)
- Исправлено соответствие оригиналу (MSE weight: 0.0 -> 0.1, loss mask)
- Обновлена документация (README.md, AGENTS.md)

---

## План повторных замеров

### 5.1 evaluate_single_model.py — Baseline

| Модель | Точность | Время/пример | Статус |
|--------|----------|--------------|--------|
| Qwen2.5-Math-1.5B-Instruct | 83.2% (1098/1319) | 7.51s | выполнено |
| Qwen3-1.7B | -- | -- | в процессе |

### 5.2 evaluate_text_mas.py — Текстовый MAS

| Метод | Точность | Время/пример | Статус |
|-------|----------|--------------|--------|
| Text MAS (3 модели) | -- | -- | ожидает |

### 6. Повторное обучение

| Этап | Роль | Статус |
|------|------|--------|
| Inner-Loop | Planner (Qwen3-1.7B) | ожидает |
| Inner-Loop | Refiner (Qwen3-1.7B) | ожидает |
| Inner-Loop | Solver (Qwen2.5-Math-1.5B) | ожидает |
| Outer-Loop | Все 3 агента | ожидает |

### 7. evaluate_recursivemas.py — RecursiveMAS

| Метод | Точность | Статус |
|-------|----------|--------|
| RecursiveMAS (latency) | -- | ожидает |

---

## Дальнейшие шаги

### Приоритет 1: Завершить повторные замеры

1. Дождаться завершения Qwen3-1.7B baseline
2. Запустить Text MAS evaluation
3. Запустить повторное обучение Inner-Loop (3 роли)
4. Запустить повторное обучение Outer-Loop
5. Запустить evaluate_recursivemas

### Приоритет 2: Ablation study

- Без Outer Link 1 (Planner -> Refiner)
- Без Outer Link 2 (Refiner -> Solver)
- Разная глубина рекурсии (r=1 vs r=3)

### Приоритет 3: Оценка на других бенчмарках

- MATH-500 (`math500`)
- GPQA (`gpqa`)
- MedQA (`medqa`)
- AIME 2025/2026 (`aime25`, `aime26`)
