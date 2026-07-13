# pregenerate_teacher_data.py — Архив

## Статус: НЕ ИСПОЛЬЗУЕТСЯ

Этот скрипт является артефактом ранней стадии разработки и **не используется** в текущей реализации.

## История создания

На раннем этапе разработки мы столкнулись с проблемой: для outer-loop обучения нужны данные с тремя полями — `question`, `plan`, `refined_plan`, `answer`. Датасет `openai/gsm8k` содержит только `question` и `answer`, поэтому возникла идея самостоятельно сгенерировать `plan` и `refined_plan` с помощью моделей Planner и Critic.

`pregenerate_teacher_data.py` был создан для:

1. Загрузки GSM8K train split (7473 примера)
2. Генерации `plan` через модель `Qwen/Qwen3-1.7B` (Planner)
3. Генерации `refined_plan` через модель `meta-llama/Llama-3.2-1B-Instruct` (Critic/Refiner)
4. Сохранения результата в JSON-файл для использования при outer-loop обучении

### Системы промптов (уникальные, не из оригинала)

Скрипт использовал авторские system prompts:

- **Planner:** "You are a math problem planner. Given a math word problem, decompose it into clear, step-by-step instructions..."
- **Critic:** "You are a math problem critic. Review the given solution plan for a math problem. Identify any gaps, errors, or missing steps..."

Эти промпты НЕ соответствуют оригинальной реализации из `mas_prompt.py`.

### Почему не используется

При приближении к оригинальной реализации мы обнаружили, что:

1. **Оригинал использует `RecursiveMAS/Sequential-Math`** — специальный датасет от авторов, который уже содержит teacher-generated `plan`, `refined_plan`, `answer`. Эти данные сгенерированы сильными моделями (Qwen3-1.7B, Llama-3.2-1B-Instruct, Qwen2.5-Math-1.5B-Instruct) с использованием оригинальных промптов из `mas_prompt.py`.

2. **Оригинал не генерирует данные локально** — он загружает готовый датасет с HuggingFace (`RecursiveMAS/Sequential-Math`, 1904 примера) и использует его напрямую.

3. **Наши промпты отличались от оригинала** — даже если бы мы сохранили скрипт, сгенерированные данные не соответствовали бы формату оригинала, потому что использовали другие system prompts.

4. **Для outer loop оригинал использует `build_stage_with_slot()`** с динамической slot injection, что невозможно предварительно токенизовать.

## Текущий подход

| Этап | Датасет | Источник |
| --- | --- | --- |
| Inner-Loop Training | `RecursiveMAS/Sequential-Math` | HuggingFace (1904 примера) |
| Outer-Loop Training | `RecursiveMAS/Sequential-Math` | HuggingFace (1904 примера) |
| Evaluation | `openai/gsm8k` (test) | HuggingFace (1319 примеров) |

### Inner loop (`train_inner.py`)

- Токенизирует весь датасет upfront перед training loop
- Использует оригинальные промпты из `mas_prompt.py`
- Поддерживает `--mas_role planner|refiner|solver`
- По умолчанию: `--dataset_name RecursiveMAS/Sequential-Math`

### Outer loop (`train_outer.py`)

- Токенизирует on-the-fly с slot injection (как в оригинале)
- Использует `build_stage_with_slot()` для injection latent embeddings
- По умолчанию: `--dataset_name RecursiveMAS/Sequential-Math`

## Артефакт сохранён для

- Исторической справки
- Понимания эволюции проекта
- Возможного будущего использования (если потребуется генерация собственных данных)
