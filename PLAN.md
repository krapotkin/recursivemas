# PLAN.md — План работ по проекту RecursiveMAS

## Текущий статус (2025-07-13)

### Выполнено

- Изучение оригинальной статьи и репозитория
- Клонирование оригинального репозитория в `experiments/original/`
- Реализация Inner-Loop Training (`train_inner.py`) — совместима с оригиналом
- Реализация Outer-Loop Training (`train_outer.py`) — полностью переписана, совместима с оригиналом
- Обучение Inner-Loop (3 роли, 20000 steps каждая)
- Обучение Outer-Loop (20000 steps)
- Оценка на GSM8K test: **81.8% accuracy (1079/1319)**

### Выполнена очистка проекта (ветка clean)

- Удалены устаревшие файлы (10 файлов)
- Чекпоинты перемещены в `~/workspace/tmp/recursivemas/old_checkpoints/`
- Удалены пустые папки `models/`, `notebooks/`
- Исправлены пути к моделям (теперь `~/workspace/models/models_llm/`)
- Убраны эмодзи из кода
- Исправлены импорты и мелкие проблемы
- Создан `.gitattributes`
- Обновлён `.gitignore`
- Обновлена документация

---

## Дальнейшие шаги

### Приоритет 1: Повторное обучение

1. Переобучить Inner-Loop (3 роли) на `RecursiveMAS/Sequential-Math`
   - Чекпоинты: `~/workspace/tmp/recursivemas/checkpoints/inner_{planner,refiner,solver}/`
2. Переобучить Outer-Loop на `RecursiveMAS/Sequential-Math`
   - Чекпоинты: `~/workspace/tmp/recursivemas/checkpoints/outer/`
3. Оценить на GSM8K test

### Приоритет 2: Тестирование

- `evaluate_single_model.py` — Qwen2.5-Math-1.5B без адаптеров
- `evaluate_text_mas.py` — текстовый MAS
- Сравнить с RecursiveMAS

### Приоритет 3: Ablation study

- Без Outer Link 1 (Planner -> Refiner)
- Без Outer Link 2 (Refiner -> Solver)
- Разная глубина рекурсии (r=1 vs r=3)

### Приоритет 4: Оценка на других бенчмарках

- MATH-500 (`math500`)
- GPQA (`gpqa`)
- MedQA (`medqa`)
- AIME 2025/2026 (`aime25`, `aime26`)
