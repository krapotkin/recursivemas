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

### Запуск процедур (шаблон)

Все процедуры запускаются с `-u` (unbuffered) и `tee` для видимых логов:

```bash
cd /home/hermes/workspace/projects/recursivemas
source .venv

# Оценка одной модели:
python3 -u custom_impl/evaluate_single_model.py \
    --model ~/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct \
    --num_examples 1319 --device cuda:2 \
    --output ~/workspace/data/recursivemas/results/eval_single_qwen25_math_15b.json \
    2>&1 | tee ~/workspace/tmp/recursivemas/eval_single_qwen25_math_15b.log

# Текстовый MAS:
python3 -u custom_impl/evaluate_text_mas.py \
    --num_examples 1319 --device cuda:2 \
    --output ~/workspace/data/recursivemas/results/eval_text_mas.json \
    2>&1 | tee ~/workspace/tmp/recursivemas/eval_text_mas.log

# Inner-Loop Training:
python3 -u custom_impl/train_inner.py \
    --model_name_or_path Qwen/Qwen3-1.7B --mas_role planner \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir ~/workspace/tmp/recursivemas/checkpoints/inner_planner \
    --max_steps 20000 --batch_size 2 --device cuda:2 \
    2>&1 | tee ~/workspace/tmp/recursivemas/train_inner_planner.log

# Outer-Loop Training:
python3 -u custom_impl/train_outer.py \
    --agent1_model Qwen/Qwen3-1.7B --agent2_model Qwen/Qwen3-1.7B \
    --agent3_model ~/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct \
    --agent1_inner ~/workspace/tmp/recursivemas/checkpoints/inner_planner/ \
    --agent2_inner ~/workspace/tmp/recursivemas/checkpoints/inner_refiner/ \
    --agent3_inner ~/workspace/tmp/recursivemas/checkpoints/inner_solver/ \
    --save_dir ~/workspace/tmp/recursivemas/checkpoints/outer/ \
    --num_recursive_rounds 3 --max_length 4096 --max_latent_tokens 80 \
    --outer_adapter_type outer_ln_res_adapter --supervise_final_only 1 \
    --batch_size 1 --grad_accum_steps 2 --max_steps 20000 --outer_lr 5e-4 \
    --device cuda:2 --dtype bfloat16 --outer_dtype bfloat16 \
    2>&1 | tee ~/workspace/tmp/recursivemas/train_outer.log

# RecursiveMAS Evaluation:
python3 -u custom_impl/evaluate_recursivemas.py \
    --agent1_model Qwen/Qwen3-1.7B --agent2_model Qwen/Qwen3-1.7B \
    --agent3_model ~/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct \
    --inner1_checkpoint ~/workspace/tmp/recursivemas/checkpoints/inner_planner/ \
    --inner2_checkpoint ~/workspace/tmp/recursivemas/checkpoints/inner_refiner/ \
    --inner3_checkpoint ~/workspace/tmp/recursivemas/checkpoints/inner_solver/ \
    --outer_checkpoint ~/workspace/tmp/recursivemas/checkpoints/outer/checkpoint-20000/ \
    --num_examples 1319 --num_latent_tokens 32 --num_recursive_rounds 3 \
    --device cuda:2 \
    --output ~/workspace/data/recursivemas/results/eval_recursivemas.json \
    2>&1 | tee ~/workspace/tmp/recursivemas/eval_recursivemas.log
```

### Текущие результаты (ветка clean)

| Метод | Точность | Время/пример | Статус |
|-------|----------|--------------|--------|
| Qwen2.5-Math-1.5B (single) | 83.2% (1098/1319) | 7.51s | выполнено |
| Qwen3-1.7B (single) | в процессе | -- | в процессе |
| Text MAS (3 модели) | -- | -- | ожидает |
| RecursiveMAS (latency) | -- | -- | ожидает |

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
