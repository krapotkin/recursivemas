# AGENTS.md — Контекст для агента RecursiveMAS

## Проект
~/workspace/projects/recursivemas — изучение и воспроизведение RecursiveMAS.

## Виртуальное окружение
- Путь: ~/workspace/venvs/recursivemas/cuda_12_x
- Python 3.12.3, PyTorch 2.6.0+cu124, transformers 4.51.0
- Активация: `source .venv` (или полный путь)

## GPU
- cuda:2 — RTX 4060 Ti 16GB (основная для обучения/eval)
- cuda:0/1 — RTX 3090 24GB

## Основные скрипты (в custom_impl/)
- `train_inner.py` — Inner-Loop обучение (planner/refiner/solver)
- `train_outer.py` — Outer-Loop обучение (с Aligned Bridge Init: --align_outer 1)
- `dataset_utils.py` — загрузка датасетов (gsm8k, math500) и парсинг ответов
- `evaluate_single_model.py` — Baseline: одна модель на GSM8K/Math500
- `evaluate_text_mas.py` — Текстовый MAS (Planner → Refiner → Solver)
- `evaluate_recursivemas.py` — RecursiveMAS с inner/outer адаптерами

## Датасеты
- Обучение: `RecursiveMAS/Sequential-Math` (1904 samples, HuggingFace)
- Оценка: `gsm8k` (1319 samples), `math500` (500 samples)

## Модели
| Роль | Модель | Hidden |
|------|--------|--------|
| Planner | Qwen3-1.7B | 1536 |
| Refiner | Qwen3-1.7B | 1536 |
| Solver | Qwen2.5-Math-1.5B-Instruct | 1536 |

## Пути
- Чекпоинты: ~/workspace/tmp/recursivemas/checkpoints/
- Модели: ~/workspace/models/models_llm/
- Результаты: ~/workspace/data/recursivemas/results/

## Чекпоинты (обучены на 20000 steps каждый)
- inner_planner/ — inner adapter для Planner
- inner_refiner/ — inner adapter для Refiner
- inner_solver/ — inner adapter для Solver
- outer/ — outer adapters (outer_12.pt, outer_23.pt, outer_adapters.pt)

## Результаты оценки
| Метод | GSM8K | Math500 |
|-------|:-----:|:-------:|
| Single Model (Qwen2.5-Math-1.5B) | 84.9% | 71.4% |
| Text MAS | 78.7% | 72.8% |
| RecursiveMAS (outer random) | 74.9% | — |
| RecursiveMAS (aligned init) | — | 76.2% |

## Aligned Bridge Init
Добавлен в train_outer.py. Перед обучением outer-loop на --align_samples (300) примерах собираются hidden states,
вычисляется ridge regression bridge, инициализируется residual_proj адаптера.
Флаги: --align_outer 1 (default=1), --align_samples 300.

## Отличия от оригинала (recursivemas-original)
- Refiner: Qwen3-1.7B вместо Llama-3.2-1B-Instruct (gated repo)
- Без accelerate: прямой PyTorch single GPU
- Датасет оценки: gsm8k + math500 (оригинал только math500)
- Aligned Bridge Init портирован из оригинального репозитория