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
- **Inner:** `R_in(h) = LayerNorm -> W2*GELU(W1*h) + h -> LayerNorm` (ln_res_adapter)
- **Outer:** `R_out(h) = LayerNorm(in) -> W2*GELU(W1*h) + W3*h -> LayerNorm(out)` (outer_ln_res_adapter)
  - `hidden_dim = out_dim * 2`
  - LayerNorm на входе и выходе
  - Residual: `Linear(in, out)`

### Training
1. **Inner loop:** cosine regression loss — align latent thoughts with Emb(ground_truth)
2. **Outer loop:** cross-entropy loss — final text output vs ground truth, gradients through all recursion rounds

### 4 collaboration patterns
- **Sequential:** Planner -> Critic -> Solver (3 agents)
- **Mixture:** Math + Code + Science -> Summarizer (4 agents)
- **Distillation:** Expert (9B) -> Learner (4B) (2 agents)
- **Deliberation:** Reflector + Tool-Caller (2 agents, external tools)

## Работа с проектом

- Временные файлы: `~/workspace/tmp/recursivemas/`
- Чекпоинты: `~/workspace/tmp/recursivemas/checkpoints/`
- Результаты оценки: `~/workspace/data/recursivemas/results/`
- Модели: `~/workspace/models/models_llm/`
- Виртуальное окружение: `~/workspace/venvs/recursivemas/cuda_12_x/`
- Документация на русском языке

### Структура custom_impl/

| Файл | Описание |
| --- | --- |
| `train_inner.py` | Inner-Loop Training (Sequential-Math, оригинальные промпты) |
| `train_outer.py` | Outer-Loop Training (полное воспроизведение оригинала) |
| `gsm8k_utils.py` | Утилиты для оценки на GSM8K |
| `evaluate_single_model.py` | Baseline: одна модель |
| `evaluate_text_mas.py` | Текстовый MAS |
| `evaluate_recursivemas.py` | RecursiveMAS оценка (оригинальные промпты) |

### Ключевые отличия от оригинала

- **Refiner модель:** `Qwen/Qwen3-1.7B` вместо `meta-llama/Llama-3.2-1B-Instruct` (gated repo)
- **Без accelerate:** используем прямой PyTorch (single GPU)
- **Претокенизация:** весь датасет токенизируется upfront в inner loop (не lazy map)
- **Дополнительные флаги:** `--grad_accum_steps`, `--load_dir`, `--enable_thinking`, `--solver_pre_question`
- **Промпты:** оригинальные из `mas_prompt.py`

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

### Стратегия оценки качества

**Уровень 1: Качество отдельных компонентов**
- Inner Link: cosine similarity между R_in(h) и Emb(ground_truth)
- Outer Link: reconstruction quality (h -> R_out(h) -> Emb(text))
- Адаптеры: grad norm, parameter norm (стабильность обучения)

**Уровень 2: Качество связки (end-to-end)**
- **Exact Match Accuracy** на GSM8K test (1319 примеров)
  - Парсим ответ из `#### X` и сравниваем с ожидаемым
- **Baseline сравнения:**
  - Single Solver (Qwen2.5-Math-1.5B без адаптеров)
  - Text-based MAS (Planner -> Critic -> Solver через текст)
  - RecursiveMAS (наша система с латентными связями)
- **Ablation:**
  - Без Outer Link 1 (Planner -> Critic)
  - Без Outer Link 2 (Critic -> Solver)
  - Разная глубина рекурсии (r=1 vs r=3)

**Метрики:**
- Accuracy (%) — доля правильно решённых задач
- Loss — CE loss на валидации
- Latency — время на один пример
- Token usage — количество сгенерированных токенов
