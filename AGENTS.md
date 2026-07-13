# AGENTS.md — Контекст для агента

## Проект RecursiveMAS

Изучение, воспроизведение и эксперименты с Recursive Multi-Agent Systems.

## Ключевые факты

- **Статья:** arXiv:2604.25917 (PDF сохранён в `~/workspace/tmp/recursivemas/paper/recursivemas.pdf`)
- **Репозиторий:** https://github.com/RecursiveMAS/RecursiveMAS
- **HF модели:** https://huggingface.co/RecursiveMAS/models
- **HF датасеты:** https://huggingface.co/RecursiveMAS/datasets

## Локальная инфраструктура

- **Сервер 192.168.45.10:** LLM API на :30000/v1 (Qwen3.6-27B-FP8) — можно использовать для LLM judge и evaluation
- **Локальная VM:** RTX 3090 x2 (ЗАНЯТЫ vLLM, НЕ ТРОГАТЬ) + **RTX 4060 Ti 16GB (cuda:2, единственная доступная)** + 188GB RAM
- **Интернет:** частичный (GitHub доступен, много timeout'ов)

## Архитектура RecursiveMAS (кратко)

### RecursiveLink
- **Inner:** `R_in(h) = LayerNorm → W2·GELU(W1·h) + h → LayerNorm` (ln_res_adapter)
- **Outer:** `R_out(h) = LayerNorm(in) → W2·GELU(W1·h) + W3·h → LayerNorm(out)` (outer_ln_res_adapter)
  - `hidden_dim = out_dim × 2`
  - LayerNorm на входе и выходе
  - Residual: `Linear(in, out)`

### Training
1. **Inner loop:** cosine regression loss — align latent thoughts with Emb(ground_truth)
2. **Outer loop:** cross-entropy loss — final text output vs ground truth, gradients through all recursion rounds

### 4 collaboration patterns
- **Sequential:** Planner → Critic → Solver (3 agents)
- **Mixture:** Math + Code + Science → Summarizer (4 agents)
- **Distillation:** Expert (9B) → Learner (4B) (2 agents)
- **Deliberation:** Reflector + Tool-Caller (2 agents, external tools)

## Работа с проектом

- Временные файлы: `~/workspace/tmp/recursivemas/`
- Виртуальное окружение: `~/workspace/venvs/recursivemas/cuda_12_x/`
- Документация на русском языке

### Кастомная реализация (Путь Б)

### Статус

- Inner-Loop Training завершён (все 3 роли, 20000 steps каждая)
- Outer-Loop Training завершён (20000 steps)
- Оценка на GSM8K (`evaluate_*.py`)
- **Результат: 81.8% accuracy (1079/1319) на GSM8K test**

### Обновление (2025-07-09) — полное воспроизведение оригинала

**train_outer.py полностью переписан** с оригинальной архитектурой:

| Компонент | Было (наше) | Стало (оригинал) |
| --- | --- | --- |
| **CrossModelAdapter** | `Linear(in, 512) → GELU → Linear(512, out) + Linear(in, out)` | `LayerNorm(in) → Linear(in, out×2) → GELU → Linear(out×2, out) + Linear(in, out) → LayerNorm(out)` |
| **hidden_dim** | 512 (фиксировано) | `out_dim × 2` (≈3072-4096) |
| **LayerNorm** | Нет | Source + Target |
| **max_length** | 256 | **4096** |
| **max_latent_tokens** | 20 | **80** |
| **num_recursive_rounds** | 1 | **3** |
| **supervise_final_only** | Нет (sum всех round'ов) | **1** (только последний round) |
| **build_stage_with_slot** | Упрощённая версия | Полное воспроизведение оригинала |
| **Loss normalization** | `total_loss / bs` | `(loss / batch_size).backward()` + gradient scaling |
| **Checkpoint format** | `outer_adapters.pt` | `outer_12.pt`, `outer_23.pt`, `outer_31.pt` + config |

**train_inner.py** практически идентичен оригиналу:
- `ln_res_adapter` (LayerNorm → MLP → residual → LayerNorm)
- Sequential-Math по умолчанию
- max_length=2048
- cosine + MSE loss
- AdamW(betas=(0.9, 0.95))
- Cosine scheduler с warmup

### Структура custom_impl/

| Файл | Статус | Описание |
| --- | --- | --- |
| `train_inner.py` | Актуален | Inner-Loop Training (Sequential-Math, оригинальные промпты) |
| `train_outer.py` | **ОБНОВЛЁН** | Outer-Loop Training (полное воспроизведение оригинала) |
| `gsm8k_utils.py` | Актуален | Утилиты для оценки на GSM8K |
| `evaluate_single_model.py` | Актуален | Baseline: одна модель |
| `evaluate_text_mas.py` | Актуален | Текстовый MAS |
| `evaluate_recursivemas.py` | Актуален | RecursiveMAS оценка (оригинальные промпты) |
| `inner_link_training.py` | УСТАРЕЛ | Заменён на `train_inner.py` |
| `outer_link_training_v2.py` | УСТАРЕЛ | Заменён на `train_outer.py` |
| `pregenerate_teacher_data.py` | УСТАРЕЛ | Артефакт, см. `pregenerate_teacher_data_ARCHIVE.md` |
| `pregenerate_teacher_data_ARCHIVE.md` | Архив | Документация об устаревшем скрипте |
| `test_pipeline.py` | Тест | Базовые тесты |
| `test_slot_injection.py` | Тест | Тест пайплайна |
| `validate.py` | Тест | Валидация |

### Ключевые отличия от оригинала (текущие)

- **Refiner модель:** Qwen3-1.7B вместо Llama-3.2-1B (gated repo)
- **Без accelerate:** используем прямой PyTorch (single GPU)
- **Претокенизация:** весь датасет токенизируется upfront в inner loop (не lazy map)
- **Дополнительные флаги:** `--grad_accum_steps`, `--load_dir`, `--enable_thinking`, `--solver_pre_question`
- **Промпты:** оригинальные из `mas_prompt.py` (обновлено 2025-07-09)

### Датасеты

| Датасет | Использование | Источник | Размер |
| --- | --- | --- | --- |
| `RecursiveMAS/Sequential-Math` | Inner + Outer training | HuggingFace | 1904 примера |
| `openai/gsm8k` (test split) | Evaluation | HuggingFace | 1319 примеров |

### Логирование (обязательно)

**Все длинные процессы (обучение, оценка) запускаются с видимыми логами.**

Правила:
1. **stdout перенаправляется в файл через `tee`** — логи видны в реальном времени И сохраняются:
   ```bash
   python3 script.py ... 2>&1 | tee ~/workspace/tmp/recursivemas/eval_live.log
   ```
2. **Лог-файлы живут в `~/workspace/tmp/recursivemas/`** с префиксом по типу:
   - `train_outer_v2.log` — обучение outer loop
   - `eval_outer_v2_live.log` — оценка в реальном времени
   - `eval_gsm8k_*.json` — результаты оценки (JSON)
3. **Процесс с `-u` (unbuffered)** — чтобы `python3` не буферизировал вывод:
   ```bash
   python3 -u script.py ...
   ```
4. **Проверка статуса через `tail -f`**:
   ```bash
   tail -f ~/workspace/tmp/recursivemas/eval_outer_v2_live.log
   ```
5. **Никогда не перезапускать процесс, не проверив что он упал.** Сначала:
   ```bash
   ps aux | grep evaluate_recursivemas | grep -v grep
   tail -5 ~/workspace/tmp/recursivemas/eval_outer_v2_live.log
   ```

### Стратегия оценки качества

**Уровень 1: Качество отдельных компонентов**
- Inner Link: cosine similarity между R_in(h) и Emb(ground_truth)
- Outer Link: reconstruction quality (h → R_out(h) → Emb(text))
- Адаптеры: grad norm, parameter norm (стабильность обучения)

**Уровень 2: Качество связки (end-to-end)**
- **Exact Match Accuracy** на GSM8K test (1319 примеров)
  - Парсим ответ из `#### X` и сравниваем с ожидаемым
- **Baseline сравнения:**
  - Single Solver (Qwen2.5-Math-1.5B без адаптеров)
  - Text-based MAS (Planner → Critic → Solver через текст)
  - RecursiveMAS (наша система с латентными связями)
- **Ablation:**
  - Без Outer Link 1 (Planner → Critic)
  - Без Outer Link 2 (Critic → Solver)
  - Разная глубина рекурсии (r=1 vs r=3)

**Метрики:**
- Accuracy (%) — доля правильно решённых задач
- Loss — CE loss на валидации
- Latency — время на один пример
- Token usage — количество сгенерированных токенов
