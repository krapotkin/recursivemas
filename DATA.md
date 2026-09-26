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
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` (HF) | 1536 | Planner, Refiner |

## Датасеты

| Датасет | Источник | Размер | Использование |
|---------|----------|--------|---------------|
| `RecursiveMAS/Sequential-Math` | HuggingFace | 1904 примера | Inner + Outer training |
| `gsm8k` (test) | HuggingFace (`openai/gsm8k`) | 1319 примеров | Evaluation |
| `math500` (test) | HuggingFace (`HuggingFaceH4/MATH-500`) | 500 примеров | Evaluation |

## Результаты оценки

| Файл | Метод | Датасет | Точность |
|------|-------|---------|----------|
| `eval_single_qwen25_math_15b_full.json` | Single Model | GSM8K | 83.2% (1098/1319) |
| `eval_text_mas_full.json` | Text MAS | GSM8K | 85.4% (1127/1319) |
| `eval_recursivemas_full.json` | RecursiveMAS (outer random) | GSM8K | 74.9% (988/1319) |
| *(новые замеры)* | Single Model | GSM8K | 84.9% (1120/1319) |
| *(новые замеры)* | Single Model | Math500 | 71.4% (357/500) |
| *(новые замеры)* | Text MAS | GSM8K | 78.7% |
| *(новые замеры)* | Text MAS | Math500 | 72.8% |

> На Math500 RecursiveMAS с aligned init показал **76.2%** (обучен в оригинальном репозитории).

## Чекпоинты

| Папка | Описание | Статус |
|-------|----------|--------|
| `inner_planner/` | Inner adapter для Planner | обучен (20000 steps) |
| `inner_refiner/` | Inner adapter для Refiner | обучен (20000 steps) |
| `inner_solver/` | Inner adapter для Solver | обучен (20000 steps) |
| `outer/` | Outer adapters (checkpoint-20000) | обучен (20000 steps) |

> Outer-адаптеры обучены без Aligned Bridge Init. Для воспроизведения результата 76.2% — используй outer адаптеры из оригинального репозитория (`recursivemas-original`).