# Сравнение: Наша реализация vs Оригинальная (RecursiveMAS)

## 🔍 Inner Link

### Оригинальная реализация (`train/model.py`)

```python
class Adapter(nn.Module):
    def __init__(self, hidden_size: int, adapter_type: str) -> None:
        adapter_type = normalize_inner_adapter_type(adapter_type)
        # Типы: linear_adapter, linear_res_adapter, adapter, res_adapter, ln_res_adapter
        
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU() if not self.is_linear else None
        self.proj2 = nn.Linear(hidden_size, hidden_size) if not self.is_linear else None
        self.use_residual = adapter_type in {"linear_res_adapter", "res_adapter", "ln_res_adapter"}
        self.pre_ln = nn.LayerNorm(hidden_size) if adapter_type == "ln_res_adapter" else None
        self.post_ln = nn.LayerNorm(hidden_size) if adapter_type == "ln_res_adapter" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_ln(x) if self.pre_ln is not None else x
        if self.is_linear:
            out = self.proj1(h)
        else:
            out = self.proj2(self.act(self.proj1(h)))
        if self.use_residual:
            out = x + out
        if self.post_ln is not None:
            out = self.post_ln(out)
        return out
```

**Дефолт:** `ln_res_adapter` (LayerNorm + 2-layer MLP + residual)

### Наша реализация (`custom_impl/inner_link_training.py`)

```python
class InnerRecursiveLink(nn.Module):
    def __init__(self, d_model):
        self.w1 = nn.Linear(d_model, d_model, bias=False)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, h):
        return h + self.w2(self.gelu(self.w1(h)))
```

**Тип:** `res_adapter` (2-layer MLP + residual, без LayerNorm)

### ✅ Различия

| Аспект | Оригинал | Наша реализация | Влияние |
|--------|-----------|-----------------|---------|
| **Дефолт тип** | `ln_res_adapter` | `res_adapter` | 🟡 Minor — LayerNorm помогает стабильности, но не критично |
| **LayerNorm** | Есть (pre + post) | Нет | 🟡 Minor — можно добавить для стабильности |
| **Bias** | Есть (по дефолту) | `bias=False` | 🟢 OK — bias не важен для MLP |
| **Loss** | `cosine + 0.1*mse` | `cosine only` | 🔴 **Major** — MSE loss помогает точности |
| **Target** | `input_embeds[:, 1:]` (каждый токен) | `mean(answer_embeddings)` (среднее по ответу) | 🔴 **Major** — оригинал учит на КАЖДОМ токене, мы на среднем |

### 📋 Критические отличия Inner Link

**1. Loss function:**
- Оригинал: `cosine_weight * cosine + mse_weight * mse`
- Мы: только `cosine`
- **Исправить:** добавить MSE компонент

**2. Target computation:**
- Оригинал: `hidden_states[:, :-1]` → `input_embeds[:, 1:]` (pairwise alignment)
- Мы: `mean(answer_embeddings)` (single vector)
- **Исправить:** использовать pairwise alignment как в оригинале

**3. Adapter type:**
- Оригинал: `ln_res_adapter` (LayerNorm + residual)
- Мы: `res_adapter` (только residual)
- **Исправить:** добавить LayerNorm для стабильности

---

## 🔍 Outer Link

### Оригинальная реализация (`train/model.py`)

```python
class CrossModelAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, adapter_type: str) -> None:
        adapter_type = normalize_outer_adapter_type(adapter_type)
        # Типы: outer_linear_adapter, outer_linear_res_adapter, outer_adapter, 
        #       outer_res_adapter, outer_ln_adapter, outer_ln_res_adapter
        
        self.is_linear = adapter_type in {"outer_linear_adapter", "outer_linear_res_adapter"}
        self.use_ln = adapter_type in {"outer_ln_adapter", "outer_ln_res_adapter"}
        self.use_residual = adapter_type in {"outer_linear_res_adapter", "outer_res_adapter", "outer_ln_res_adapter"}

        hidden_dim = out_dim * 2 if self.use_ln else out_dim  # ← expansion!
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU() if not self.is_linear else None
        self.proj2 = nn.Linear(hidden_dim, out_dim) if not self.is_linear else None

        self.ln_source = nn.LayerNorm(in_dim) if self.use_ln else None
        self.ln_target = nn.LayerNorm(out_dim) if self.use_ln else None
        self.residual_proj = nn.Linear(in_dim, out_dim) if self.use_residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_source(x) if self.ln_source is not None else x
        if self.is_linear:
            out = self.proj1(h)
        else:
            out = self.proj2(self.act(self.proj1(h)))
        if self.residual_proj is not None:
            out = out + self.residual_proj(x)
        if self.ln_target is not None:
            out = self.ln_target(out)
        return out
```

**Дефолт:** `outer_ln_res_adapter` (LayerNorm + expansion 2x + residual + LayerNorm)

### Наша реализация (`custom_impl/outer_link_training.py`)

```python
class OuterRecursiveLink(nn.Module):
    def __init__(self, d_in, d_out):
        self.w1 = nn.Linear(d_in, d_out, bias=False)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(d_out, d_out, bias=False)
        self.w3 = nn.Linear(d_in, d_out, bias=False)  # для маппинга размеров
    
    def forward(self, h):
        return self.w3(h) + self.w2(self.gelu(self.w1(h)))
```

**Тип:** `outer_res_adapter` (2-layer MLP + residual, без LayerNorm, без expansion)

### ✅ Различия

| Аспект | Оригинал | Наша реализация | Влияние |
|--------|-----------|-----------------|---------|
| **Дефолт тип** | `outer_ln_res_adapter` | `outer_res_adapter` | 🟡 Minor |
| **LayerNorm** | Есть (source + target) | Нет | 🟡 Minor — помогает стабильности |
| **Hidden dim** | `out_dim * 2` (expansion) | `out_dim` (no expansion) | 🔴 **Major** — expansion даёт больше capacity |
| **Residual** | `residual_proj(x)` | `w3(h)` | 🟢 OK — функционально одинаково |
| **Bias** | Есть | `bias=False` | 🟢 OK |

### 📋 Критические отличия Outer Link

**1. Hidden dimension:**
- Оригинал: `hidden_dim = out_dim * 2` (expansion 2x)
- Мы: `hidden_dim = out_dim` (no expansion)
- **Исправить:** добавить expansion для capacity

**2. LayerNorm:**
- Оригинал: `LayerNorm(in_dim)` → MLP → `LayerNorm(out_dim)`
- Мы: нет LayerNorm
- **Исправить:** добавить LayerNorm для стабильности

---

## 🔍 Training Pipeline

### Inner Loop

| Аспект | Оригинал | Наша реализация |
|--------|-----------|-----------------|
| **Framework** | Accelerate (DDP support) | Чистый PyTorch |
| **Loss** | `cosine + 0.1*mse` | `cosine only` |
| **Target** | Pairwise: `hidden[:, :-1]` → `embeds[:, 1:]` | Mean: `mean(answer_embeds)` |
| **Optimizer** | AdamW(betas=(0.9, 0.95)) | AdamW(default betas) |
| **Scheduler** | Cosine with warmup | Cosine Annealing |
| **Grad accum** | Есть | Нет |
| **Gradient clipping** | Есть (1.0) | Нет |

### Outer Loop

| Аспект | Оригинал | Наша реализация |
|--------|-----------|-----------------|
| **Data format** | `{question, plan, refined_plan, answer}` | `{question, answer}` |
| **Prompting** | Role-specific prompts с slots | Простые role prompts |
| **Latent injection** | Встраивает латентные векторы в prompt через slot | Concatenates role + question |
| **Loss** | CE loss на solver output | CE loss на prediction head |
| **Supervision** | Final round only или weighted | Final round only |
| **Latent steps** | 80 (default) | 3 (для скорости) |
| **Recursion rounds** | 3 (default) | 2 |
| **Gradient flow** | Через все round'ы (unrolled) | Через все round'ы (unrolled) ✅ |

---

## 📊 Итоговая оценка

### Что мы сделали ПРАВИЛЬНО ✅

1. **Архитектура RecursiveLink** — правильная (MLP + residual)
2. **Inner-Outer loop separation** — правильная
3. **Frozen base models** — правильно
4. **Unrolled recursion** — правильно (градиенты через все round'ы)
5. **Cross-model adapter** — правильная концепция (W3 для dimension mapping)
6. **Cosine similarity loss** для inner loop — правильно

### Что нужно ИСПРАВИТЬ 🔧

1. **Inner Link Loss:** добавить MSE компонент (`cosine + 0.1*mse`)
2. **Inner Link Target:** использовать pairwise alignment вместо mean
3. **Outer Link Expansion:** добавить 2x expansion (`hidden_dim = out_dim * 2`)
4. **LayerNorm:** добавить в Inner и Outer Link для стабильности
5. **Optimizer:** использовать `betas=(0.9, 0.95)` как в оригинале
6. **Gradient clipping:** добавить (max_norm=1.0)
7. **Latent steps:** увеличить до 80 (или хотя бы 20 для демо)
8. **Data format:** использовать `{question, plan, refined_plan, answer}` для outer loop

### Что можно ПРОИГНОРИРОВАТЬ 🟡

1. **Accelerate framework** — для обучения концепции не нужно
2. **DDP support** — у нас одна GPU
3. **Gradient accumulation** — не критично для демо
4. **Role-specific prompts** — можно упростить

---

## 🎯 Приоритетные исправления

**Критичные (влияют на корректность):**
1. Inner Link target computation (pairwise vs mean)
2. Outer Link hidden dimension (expansion 2x)
3. Inner Link loss (добавить MSE)

**Рекомендуемые (влияют на качество):**
4. LayerNorm в Inner и Outer Link
5. Optimizer betas=(0.9, 0.95)
6. Gradient clipping

**Опциональные (для production):**
7. Accelerate framework
8. Gradient accumulation
9. Role-specific prompts
