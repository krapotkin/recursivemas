# 📚 План изучения RecursiveMAS

> Поэтапный план от теории к собственным экспериментам.
> Каждый этап — самостоятельная веха с конкретными результатами.

---

## 🗺️ Обзор маршрута

```
Этап 0: Фундамент (теория)
    ↓
Этап 1: Разведка (репо + чужой код)
    ↓
Этап 2: Первый запуск (инференс чекпоинтов)
    ↓
Этап 3: Воспроизведение (training pipeline)
    ↓
Этап 4: Глубокий анализ (ablation + визуализация)
    ↓
Этап 5: Собственные эксперименты
    ↓
Этап 6: Продвинутые модификации
```

---

## Этап 0 — Фундамент: Теория и концепции

**Цель:** Понять *что*, *почему* и *зачем* в RecursiveMAS.

### 0.1 — Предварительные знания
- [ ] **Рекурсивные языковые модели (RLM)** — LoopLM, ReCurrent, и другие подходы к итеративному уточнению в латентном пространстве
- [ ] **Мульти-агентные системы (MAS)** — Chain-of-Agents, Mixture-of-Agents, TextGrad, и паттерны коллаборации
- [ ] **Transformer internals** — embedding layer, hidden states, LM head, residual stream
- [ ] **Fine-tuning paradigms** — LoRA, full SFT, PEFT

### 0.2 — Глубокое чтение статьи
- [ ] Прочитать статью полностью, разбирая каждый раздел
- [ ] Разобрать **Figure 2** (архитектура) — как данные проходят через систему
- [ ] Разобрать **Figure 3** (RecursiveLink) — inner vs outer link
- [ ] Разобрать **Figure 4** (training pipeline) — inner loop → outer loop
- [ ] Понять **Proposition 3.1** (runtime complexity) — почему латентное пространство эффективнее
- [ ] Понять **Theorem 4.1** (gradient stability) — почему градиенты не затухают
- [ ] Разобрать **Table 1** (модели) — какие модели используются для каждого стиля
- [ ] Разобрать **Table 2** (результаты) — как меняется точность/скорость/токены с глубиной рекурсии

### 0.3 — Конспекты в `theory/`
- [ ] `theory/01_architecture.md` — архитектура RecursiveMAS своими словами
- [ ] `theory/02_recursive_link.md` — математика RecursiveLink (inner + outer)
- [ ] `theory/03_training.md` — inner-outer loop training, loss functions
- [ ] `theory/04_collaboration_patterns.md` — 4 стиля коллаборации
- [ ] `theory/05_theory.md` — теоретические гарантии (complexity + gradient stability)

**Результат этапа:** Ты можешь объяснить RecursiveMAS другу за 10 минут.

---

## Этап 1 — Разведка: Изучение репозитория

**Цель:** Понять код, структуру, зависимости.

### 1.1 — Клонирование и окружение
- [ ] Склонировать репозиторий в `experiments/original/`
- [ ] Создать venv: `~/workspace/venvs/recursivemas/cuda_12_x/`
- [ ] Установить зависимости из `requirements.txt`
- [ ] Проверить, что CUDA доступна

### 1.2 — Код-ревью
- [ ] **`train/model.py`** — реализация RecursiveLink (inner + outer)
- [ ] **`train/train_inner.py`** — inner loop training pipeline
- [ ] **`train/train_outer.py`** — outer loop training pipeline
- [ ] **`train/outer/sequential.py`** — sequential collaboration logic
- [ ] **`inference/modeling.py`** — inference-time RecursiveLink
- [ ] **`inference/inference_utils/inference_mas.py`** — inference pipeline
- [ ] **`inference/system_loader.py`** — загрузка системы из чекпоинтов

### 1.3 — Заметки по коду
- [ ] `theory/06_code_review.md` — карта кода, как всё связано
- [ ] `theory/07_data_format.md` — формат данных для обучения (HF datasets)

**Результат этапа:** Ты понимаешь, как работает каждый файл, и можешь запустить любой скрипт.

---

## Этап 2 — Первый запуск: Инференс

**Цель:** Запустить готовую систему на реальных данных и увидеть результат.

### 2.1 — Plug-and-play с HF чекпоинтами
> ⚠️ **Ограничение:** только GPU #2 (RTX 4060 Ti 16GB). Используем `--device cuda:2`.
> Только **Sequential Light** (~4.2B суммарно) влезет в 16GB VRAM. Sequential Scaled (~11B) — нет.

- [ ] Скачать чекпоинты Sequential Light (~4.2B суммарно)
- [ ] Запустить `inference/run.py --style sequential_light --dataset math500 --device cuda:2`
- [ ] Запустить `inference/run.py --style sequential_light --dataset gpqa --device cuda:2`
- [ ] Сравнить результаты с Table 2 из статьи

### 2.2 — Сравнение стилей
> ⚠️ С 16GB VRAM только Sequential Light реалистичен. Mixture (7B+), Distillation (13B+), Deliberation (8B+) не влезут.
> Альтернатива: запускать инференс через API на 192.168.45.10 (Qwen3.6-27B) для сравнения.

- [ ] ~Sequential Scaled (если влезает в VRAM)~ **Пропуск: ~11B не влезает в 16GB**
- [ ] ~Запустить Distillation style~ **Пропуск: ~13B не влезает**
- [ ] ~Запустить Mixture style~ **Пропуск: ~7B+ не влезает**
- [ ] Сравнить результаты Sequential Light с baseline (single agent)
- [ ] Сравнить точность, скорость, использование токенов

### 2.3 — Визуализация
- [ ] Ноутбук: `notebooks/01_inference_results.ipynb` — графики accuracy по round'ам
- [ ] Ноутбук: `notebooks/02_latency_analysis.ipynb` — анализ скорости и токенов
- [ ] Ноутбук: `notebooks/03_sample_outputs.ipynb` — качественные примеры ответов

**Результат этапа:** Рабочая система, результаты воспроизведены, есть графики.

---

## Этап 3 — Воспроизведение: Training Pipeline

**Цель:** Обучить свою систему с нуля, следуя оригинальной реализации.

> **🔄 ОБНОВЛЕНО (2025):** Переход от GSM8K-прототипа к воспроизведению оригинала.
> Теперь используем `RecursiveMAS/Sequential-Math` из HuggingFace и оригинальные промпты.

### 3.1 — Подготовка данных
- [x] Изучить HF datasets: `RecursiveMAS/Sequential-Math`, `RecursiveMAS/Sequential-Code`
- [x] Понять формат данных (question, plan, refined_plan, answer)
- [x] Понять оригинальные промпты (`mas_prompt.py`)
- [ ] Ноутбук: `notebooks/04_data_exploration.ipynb` — EDA тренировочных данных

**Датасет:** `RecursiveMAS/Sequential-Math` (HuggingFace, 1904 примера)
- Содержит: `question`, `plan`, `refined_plan`, `answer`, `answer_field`
- Данные сгенерированы авторами статьи с помощью мощных моделей-учителей
- Используется для Inner-Loop и Outer-Loop training

**⚠️ Важно:** Мы НЕ используем GSM8K для обучения. GSM8K используется только для оценки (evaluation).
Оригинальная реализация обучает на `RecursiveMAS/Sequential-Math` и оценивает на `math500`.
Мы оцениваем на GSM8K test (1319 примеров) для проверки генерализации.

### 3.2 — Inner Loop Training
> ⚠️ Inner loop обучает по одному агенту — каждый влезает в 16GB отдельно.
> Стратегия: **последовательное обучение**, не одновременно.
>
> **Скрипт:** `custom_impl/train_inner.py`
>
> **🔄 ТРЕБУЕТСЯ ПЕРЕОБУЧЕНИЕ (2025-07-09):** Промпты обновлены до оригинальных из `mas_prompt.py`.
> Старые чекпоинты обучены на модифицированных промптах и будут заменены.

```bash
# Planner (Qwen3-1.7B)
python custom_impl/train_inner.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --mas_role planner \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir custom_impl/checkpoints/inner_planner \
    --max_steps 20000 \
    --batch_size 2 \
    --adapter_lr 5e-4 \
    --warmup_steps 10 \
    --log_every 50 \
    --save_steps 1000 \
    --device cuda:2 \
    --dtype bfloat16

# Refiner/Critic (Qwen3-1.7B, вместо Llama-3.2-1B — gated repo)
python custom_impl/train_inner.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --mas_role refiner \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir custom_impl/checkpoints/inner_refiner \
    --max_steps 20000 \
    --batch_size 2 \
    --adapter_lr 5e-4 \
    --warmup_steps 10 \
    --log_every 50 \
    --save_steps 1000 \
    --device cuda:2 \
    --dtype bfloat16

# Solver (Qwen2.5-Math-1.5B-Instruct)
python custom_impl/train_inner.py \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
    --mas_role solver \
    --dataset_name RecursiveMAS/Sequential-Math \
    --save_dir custom_impl/checkpoints/inner_solver \
    --max_steps 20000 \
    --batch_size 2 \
    --adapter_lr 5e-4 \
    --warmup_steps 10 \
    --log_every 50 \
    --save_steps 1000 \
    --device cuda:2 \
    --dtype bfloat16
```

**Результаты предыдущего обучения (на модифицированных промптах):**

| Роль | Модель | Final Loss | Время | Статус |
|------|--------|-----------|-------|--------|
| **Planner** | Qwen3-1.7B | **0.254** (cos) | ~30 мин | ✅ (требует переобучения) |
| **Refiner** | Qwen3-1.7B | **0.245** (cos) | ~46 мин | ✅ (требует переобучения) |
| **Solver** | Qwen2.5-Math-1.5B | **0.171** (cos) | ~79 мин | ✅ (требует переобучения) |

- [ ] Переобучить inner link для Planner (оригинальные промпты)
- [ ] Переобучить inner link для Refiner (оригинальные промпты)
- [ ] Переобучить inner link для Solver (оригинальные промпты)
- [ ] Проверить, что loss сходится (все три роли: стабильное падение)

**Примечание:** Refiner обучен на Qwen3-1.7B вместо Llama-3.2-1B-Instruct, потому что Llama — gated repo без HF_TOKEN. Это не критично для pipeline, потому что hidden size одинаковый (2048).

**Параметры (как в оригинале):**
- Adapter: `ln_res_adapter` (LayerNorm → MLP → residual → LayerNorm)
- Loss: cosine similarity (weight=1.0) + MSE (weight=0.0)
- Optimizer: AdamW(betas=(0.9, 0.95))
- Scheduler: cosine with warmup (10 steps)
- LR: 5e-4, max_steps: 20000
- **Prompts:** оригинальные из `mas_prompt.py` (обновлено 2025-07-09)

### 3.3 — Outer Loop Training
> ⚠️ **Ограничение: 1x RTX 4060 Ti 16GB.**
> Стратегия: **последовательное обучение** (inner → outer), не одновременно.
>
> **Скрипт:** `custom_impl/train_outer.py`
>
> **🔄 ОБНОВЛЕНО (2025-07-09):** Полное воспроизведение оригинальной архитектуры.

```bash
python custom_impl/train_outer.py \
    --agent1_model Qwen/Qwen3-1.7B \
    --agent2_model Qwen/Qwen3-1.7B \
    --agent3_model ./models/Qwen2.5-Math-1.5B-Instruct \
    --agent1_inner custom_impl/checkpoints/inner_planner/ \
    --agent2_inner custom_impl/checkpoints/inner_refiner/ \
    --agent3_inner custom_impl/checkpoints/inner_solver/ \
    --save_dir custom_impl/checkpoints/outer/ \
    --num_recursive_rounds 3 \
    --max_length 4096 \
    --max_latent_tokens 80 \
    --outer_adapter_type outer_ln_res_adapter \
    --supervise_final_only 1 \
    --batch_size 2 \
    --max_steps 20000 \
    --outer_lr 5e-4 \
    --device cuda:2
```

> **VRAM бюджет:**
> - Planner (Qwen3-1.7B): ~3.4GB BF16
> - Refiner (Qwen3-1.7B): ~3.4GB BF16 (shared с Planner если та же модель)
> - Solver (Qwen2.5-Math-1.5B): ~3GB BF16
> - Outer Link адаптеры: ~35MB (3 × ~11.8M params)
> - Итого: ~10GB + optimizer/gradients = ~14-16GB ⚠️ (требует оптимизации)

**Параметры (как в оригинале):**
- Outer adapter: `outer_ln_res_adapter` (LayerNorm → Linear(in, out×2) → GELU → Linear(out×2, out) + Linear(in, out) → LayerNorm(out))
- Loss: CE на solver output, `supervise_final_only=1` (только последний recursive round)
- Optimizer: AdamW(betas=(0.9, 0.95))
- Scheduler: cosine with warmup (10 steps)
- LR: 5e-4, max_steps: 20000
- **max_length: 4096** (было 256)
- **max_latent_tokens: 80** (было 20)
- **num_recursive_rounds: 3** (было 1)

- [ ] Собрать систему из inner-trained агентов
- [ ] Запустить outer loop training
- [ ] Мониторить loss и accuracy на валидации
- [ ] Сохранить чекпоинты

### 3.4 — Оценка (GSM8K)
> **Важно:** обучение на Sequential-Math, оценка на GSM8K test.
> Это позволяет проверить генерализацию на другой датасет.
>
> **📊 РЕЗУЛЬТАТ (2025-07-09, на старых чекпоинтах):**
> - **78.2% accuracy (1031/1319)** на GSM8K test (полный датасет)
> - Pipeline: Planner → inner_1 → outer_12 → Refiner → inner_2 → outer_23 → Solver
> - Среднее время на пример: 8.78s
>
> **⚠️ После переобучения с оригинальными промптами результат может измениться.**
>
> **Скрипты оценки (без изменений):**
> - `evaluate_single_model.py` — baseline
> - `evaluate_text_mas.py` — текстовый MAS
> - `evaluate_recursivemas.py` — наша система (обновлён для работы с train_outer.py)

### 3.5 — Кастомная реализация (история)
> ⚠️ **Устаревшие скрипты:** `inner_link_training.py`, `outer_link_training_v2.py`, `pregenerate_teacher_data.py`
> Сохранены как исторические артефакты. Используют GSM8K + синтетические данные.
> Заменены на `train_inner.py` и `train_outer.py` (Sequential-Math).
>
> **📋 Архивная документация:** `custom_impl/pregenerate_teacher_data_ARCHIVE.md`
> Описывает, как появился `pregenerate_teacher_data.py`, почему мы его не используем и что заменило.
>
> **✅ Обновлён:** `evaluate_recursivemas.py` — теперь использует полный pipeline из `train_outer.py`:
> - Planner → inner_1 → latent tokens → outer_12 → Refiner → inner_2 → latent tokens → outer_23 → Solver
> - Autoregressive latent token generation через inner adapters
> - Slot injection как в оригинале (`<<LATENT_PLANNER_SLOT>>`, `<<LATENT_REFINED_SLOT>>`)
> - CrossModelAdapter (тот же класс, что в `train_outer.py`)
> - Правильный парсинг GSM8K (`#### X` формат)

**Что было реализовано в прототипе:**
- ✅ Slot injection через placeholder
- ✅ Teacher forcing (hidden states от assistant токенов)
- ✅ Chat templates (apply_chat_template)
- ✅ CE loss на ВСЕХ токенах ответа
- ✅ Dimension mapping (2048 → 1536)
- ✅ Backward pass (градиенты проходят через оба адаптера)

### 3.6 — Стратегия оценки качества
> **Два уровня оценки:**
>
> **Уровень 1: Качество отдельных компонентов**
> - Inner Link: cosine similarity между R_in(h) и Emb(ground_truth)
> - Outer Link: reconstruction quality (h → R_out(h) → Emb(text))
> - Адаптеры: grad norm, parameter norm (стабильность обучения)
>
> **Уровень 2: Качество связки (end-to-end)**
> - **Exact Match Accuracy** на GSM8K test (1319 примеров)
>   - Парсим ответ из `#### X` и сравниваем с ожидаемым
> - **Baseline сравнения:**
>   - Single Solver (Qwen2.5-Math-1.5B без адаптеров)
>   - Text-based MAS (Planner → Critic → Solver через текст)
>   - RecursiveMAS (наша система с латентными связями)
> - **Ablation:**
>   - Без Outer Link 1 (Planner → Critic)
>   - Без Outer Link 2 (Critic → Solver)
>   - Разная глубина рекурсии (r=1 vs r=2)
>
> **Метрики:**
> - Accuracy (%) — доля правильно решённых задач
> - Loss — CE loss на валидации
> - Latency — время на один пример
> - Token usage — количество сгенерированных токенов

**Результат этапа:** Собственная обученная система, результаты сравнимы со статьёй.

---

## Этап 4 — Глубокий анализ

**Цель:** Понять *почему* это работает, а не просто *что* работает.

### 4.1 — Ablation studies
- [ ] **RecursiveLink design** — 1-layer vs 2-layer, с residual vs без (Table 4)
- [ ] **Глубина рекурсии** — r=1 vs r=2 vs r=3 (Table 2)
- [ ] **Длина латентных мыслей** — m=20 vs m=40 vs m=80 (Figure 8)
- [ ] **Inner vs outer только** — что даёт каждый компонент

### 4.2 — Визуализация латентного пространства
- [ ] PCA/UMAP проекция латентных мыслей по round'ам
- [ ] Косинусное сходство между round'ами (как меняется)
- [ ] Семантическое распределение (как в Figure 7 статьи)

### 4.3 — Анализ градиентов
- [ ] Нормы градиентов по round'ам (проверка Theorem 4.1)
- [ ] Сравнение с text-based baseline

### 4.4 — Ноутбуки
- [ ] `notebooks/05_latent_space_viz.ipynb` — визуализация латентных представлений
- [ ] `notebooks/06_gradient_analysis.ipynb` — анализ градиентов
- [ ] `notebooks/07_ablation.ipynb` — ablation experiments

**Результат этапа:** Глубокое понимание механизмов, визуализации, графики.

---

## Этап 5 — Собственные эксперименты

**Цель:** Применить RecursiveMAS к новым задачам и настройкам.

### 5.1 — Новые модели
- [ ] Попробовать с локальными моделями (Qwen3.5-0.8B на 192.168.45.10 — идеально для 16GB VRAM)
- [ ] Попробовать минимальную конфигурацию: 3x Qwen3.5-0.8B (~2.4GB суммарно, легко влезает)
- [ ] Сравнить эффективность разных backbone'ов

### 5.2 — Новые задачи
- [ ] Добавить русскоязычные задачи (если возможно)
- [ ] Попробовать на задачах из нашей области
- [ ] Кастомный датасет для специфической задачи

### 5.3 — Оптимизация
- [ ] Quantization (INT4/INT8) для inference
- [ ] vLLM serving для ускорения
- [ ] Оптимизация памяти для outer loop training

### 5.4 — Кастомная реализация
- [ ] `custom_impl/` — собственная минимальная реализация RecursiveLink
- [ ] Упрощённый pipeline для экспериментов
- [ ] Интеграция с нашим local-agent-bench

**Результат этапа:** Собственные результаты, кастомная реализация, оптимизации.

---

## Этап 6 — Продвинутые модификации

**Цель:** Улучшить или расширить оригинальный подход.

### 6.1 — Архитектурные модификации
- [ ] Attention-based RecursiveLink (вместо MLP)
- [ ] Adaptive recursion depth (динамическое число round'ов)
- [ ] Cross-attention между агентами

### 6.2 — Training improvements
- [ ] Curriculum learning для outer loop
- [ ] Gradient clipping strategies
- [ ] Multi-task outer loop training

### 6.3 — Интеграция с существующими системами
- [ ] RecursiveMAS + RAG
- [ ] RecursiveMAS + tool-use (как Deliberation style, но расширенный)
- [ ] RecursiveMAS как замена chain-of-agents в существующих проектах

### 6.4 — Публикация результатов
- [ ] `notebooks/08_custom_results.ipynb` — собственные результаты
- [ ] `theory/08_lessons_learned.md` — выводы и рекомендации

---

## ⏱️ Оценочные сроки

| Этап | Сложность | Ориентировочно |
|------|-----------|----------------|
| 0 — Фундамент | 🟢 Низкая | 1–2 дня |
| 1 — Разведка | 🟢 Низкая | 1–2 дня |
| 2 — Инференс | 🟡 Средняя | 2–3 дня |
| 3 — Training | 🔴 Высокая | 5–7 дней |
| 4 — Анализ | 🟡 Средняя | 3–5 дней |
| 5 — Эксперименты | 🟡 Средняя | 5–7 дней |
| 6 — Модификации | 🔴 Высокая | 7–14 дней |

**Итого:** ~3–5 недель при активном изучении.

---

## 🧠 Полезные ссылки

- [Статья на arXiv](https://arxiv.org/abs/2604.25917)
- [PDF статьи](https://arxiv.org/pdf/2604.25917)
- [Репозиторий GitHub](https://github.com/RecursiveMAS/RecursiveMAS)
- [HuggingFace модели](https://huggingface.co/RecursiveMAS/models)
- [HuggingFace датасеты](https://huggingface.co/RecursiveMAS/datasets)
- [Project website](https://recursivemas.github.io/)
- [YouTube tutorial](https://www.youtube.com/watch?v=dUmT0OIGoqE)
- [Interactive demo](https://vishalmysore.github.io/recursiveMASDemo/)
- [VentureBeat статья](https://venturebeat.com/ai/how-recursivemas-speeds-up-multi-agent-inference-by-2-4x-and-reduces-token-usage-by-75)

### Смежные работы
- **LoopLM** (Zhu et al., 2025) — рекурсивные языковые модели
- **TextGrad** (Yuksekgonul et al., 2025) — текстовые градиенты для MAS
- **Mixture-of-Agents** (Wang et al., 2025) — MoA для LLM
