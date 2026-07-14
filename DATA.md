# DATA.md — Описание данных

## Расположение

| Тип | Путь | Описание |
|-----|------|----------|
| Модели | `~/workspace/models/models_llm/` | Скачанные LLM модели |
| Чекпоинты | `~/workspace/tmp/recursivemas/checkpoints/` | Обученные адаптеры (inner/outer) |
| Результаты | `~/workspace/data/recursivemas/results/` | JSON файлы с результатами оценки |
| Логи | `~/workspace/tmp/recursivemas/` | Лог-файлы обучения и оценки |

## Модели

| Модель | Путь | Hidden size | Роль |
|--------|------|-------------|------|
| Qwen2.5-Math-1.5B-Instruct | `~/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct/` | 1536 | Solver |
| Qwen3-1.7B | `~/workspace/models/models_llm/Qwen3-1.7B/` | 2048 | Planner, Refiner |

## Датасеты

| Датасет | Источник | Размер | Использование |
|---------|----------|--------|---------------|
| `RecursiveMAS/Sequential-Math` | HuggingFace | 1904 примера | Inner + Outer training |
| `openai/gsm8k` (test) | HuggingFace | 1319 примеров | Evaluation |

## Результаты оценки

| Файл | Метод | Точность | Дата |
|------|-------|----------|------|
| `eval_single_qwen25_math_15b_full.json` | Qwen2.5-Math-1.5B (single) | 83.2% (1098/1319) | 2025-07-13 |
| `eval_single_qwen3_17b_full.json` | Qwen3-1.7B (single) | 22.3% (294/1319) | 2025-07-13 |
| `eval_text_mas_full.json` | Text MAS (3 модели) | 85.4% (1127/1319) | 2025-07-13 |
| `eval_recursivemas.json` | RecursiveMAS (latency) | -- | ожидает |

## Чекпоинты

| Папка | Описание | Статус |
|-------|----------|--------|
| `inner_planner/` | Inner adapter для Planner | обучен (старый) |
| `inner_refiner/` | Inner adapter для Refiner | обучен (старый) |
| `inner_solver/` | Inner adapter для Solver | обучен (старый) |
| `outer/` | Outer adapters (checkpoint-20000) | обучен (старый) |

Примечание: старые чекпоинты сохранены в `~/workspace/tmp/recursivemas/old_checkpoints/`.
