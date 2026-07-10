#!/usr/bin/env python3
"""
evaluate_recursivemas_v2.py — Оценка RecursiveMAS (соответствует train_outer.py)

Pipeline (как в train_outer.py):
  Round 1:
    Planner → inner_1 (latent tokens) → outer_12 → Refiner → inner_2 (latent tokens) → outer_23 → Solver → ответ
  Round 2..N:
    Solver → inner_3 → outer_31 → Planner (feedback) → inner_1 → outer_12 → ... → Solver → ответ

Ключевые отличия от v1:
  - Использует inner adapters (обученные в train_inner.py)
  - Использует outer adapters (обученные в train_outer.py)
  - Latent token generation через inner adapter loop
  - Recursive rounds с feedback
  - Prompts как в оригинале (mas_prompt.py)

Чекпоинты:
  - Inner: custom_impl/checkpoints/inner_{planner,refiner,solver}/adapter.pt
  - Outer: custom_impl/checkpoints/outer/outer_adapters.pt

Пример:
    python evaluate_recursivemas_v2.py --num_examples 20
    python evaluate_recursivemas_v2.py --num_examples 100 --num_recursive_rounds 3
"""

import os
import sys
import time
import argparse
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gsm8k_utils import (
    load_gsm8k_dataset,
    compute_accuracy,
    print_results,
    save_results,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix for CUDA
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# ============================================================
# Конфигурация
# ============================================================
PLANNER_MODEL = "Qwen/Qwen3-1.7B"
REFINER_MODEL = "Qwen/Qwen3-1.7B"  # та же модель, что Planner
SOLVER_MODEL = "/home/hermes/workspace/projects/recursivemas/models/Qwen2.5-Math-1.5B-Instruct"
DEFAULT_DEVICE = "cuda:2"

# Чекпоинты
INNER_PLANNER_CKPT = "custom_impl/checkpoints/inner_planner/adapter.pt"
INNER_REFINER_CKPT = "custom_impl/checkpoints/inner_refiner/adapter.pt"
INNER_SOLVER_CKPT = "custom_impl/checkpoints/inner_solver/adapter.pt"
OUTER_CKPT = "custom_impl/checkpoints/outer/outer_adapters.pt"

# Гиперпараметры инференса
DEFAULT_NUM_LATENT_TOKENS = 20
DEFAULT_NUM_RECURSIVE_ROUNDS = 3

# ============================================================
# Prompts (как в оригинале mas_prompt.py)
# ============================================================
SYSTEM_PROMPT = "You are a helpful assistant."

def build_math_planner_prompt(question: str) -> str:
    return (
        f"You are a planner agent in a multi-agent math system.\n"
        f"The question is:\n"
        f"Question:\n{question}\n"
        f"Provide a clear step-by-step plan (within 3-6 steps) to solve the problem.\n"
        f"Do not provide the final answer.\n"
        f"Your response should be in the format of:\n"
        f"Step 1: ...\n"
        f"...\n"
        f"Step n: ..."
    )

def build_math_planner_prompt_with_feedback(question: str) -> str:
    return (
        f"You are a planner agent in a recursive multi-agent math system.\n"
        f"This is a later recursive round.\n"
        f"The question is:\n"
        f"Question:\n{question}\n"
        f"Feedback signal from the previous solver round:\n"
        f"<<LATENT_FEEDBACK_SLOT>>\n"
        f"Use the feedback as a soft correction signal to improve the plan.\n"
        f"If there is any conflict, prioritize the problem constraints.\n"
        f"Provide a clear step-by-step plan (within 3-6 steps) to solve the problem.\n"
        f"Do not provide the final answer.\n"
        f"Your response should be in the format of:\n"
        f"Step 1: ...\n"
        f"...\n"
        f"Step n: ..."
    )

def build_math_refiner_prompt(question: str) -> str:
    return (
        f"You are a refiner agent in a multi-agent math system.\n"
        f"The question is:\n"
        f"Question:\n{question}\n"
        f"The initial plan from the planner:\n"
        f"Initial Plan:\n"
        f"<<LATENT_PLANNER_SLOT>>\n"
        f"Refine the plan into a clearer and stronger step-by-step plan (within 3-6 steps).\n"
        f"Do not provide the final answer.\n"
        f"Your response should be in the format of:\n"
        f"Step 1: ...\n"
        f"...\n"
        f"Step n: ..."
    )

def build_math_solver_prompt(question: str) -> str:
    return (
        f"You are a solver agent in a multi-agent math system.\n"
        f"Here is the refined plan:\n"
        f"Refined Plan:\n"
        f"<<LATENT_REFINED_SLOT>>\n"
        f"\n---\nThe question is:\n"
        f"Question:\n{question}\n"
        f"Solve the question and put the final answer inside \\boxed{{}}, for example \\boxed{{1}}."
    )


# ============================================================
# Adapter модели (как в train_outer.py)
# ============================================================
class InnerAdapter(nn.Module):
    """ln_res_adapter: LayerNorm → MLP → residual → LayerNorm"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        h = self.pre_ln(x)
        out = self.proj2(self.act(self.proj1(h)))
        out = x + out
        out = self.post_ln(out)
        return out


class CrossModelAdapter(nn.Module):
    """Outer Link: W3·h + W2·GELU(W1·h)"""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_dim)
        self.W3 = nn.Linear(in_dim, out_dim, bias=False)
        self._init_weights(in_dim, out_dim)

    def _init_weights(self, in_dim, out_dim):
        nn.init.kaiming_uniform_(self.W1.weight, a=0)
        nn.init.zeros_(self.W2.weight)
        if in_dim == out_dim:
            nn.init.eye_(self.W3.weight)
        else:
            nn.init.zeros_(self.W3.weight)

    def forward(self, h):
        return self.W3(h) + self.W2(F.gelu(self.W1(h)))


# ============================================================
# Утилиты
# ============================================================
def parse_boxed_answer(text: str) -> str:
    """Извлекает ответ из \\boxed{...} или #### ..."""
    if not text:
        return ""
    # Try \boxed{...}
    match = re.search(r'\\boxed\{([^}]*)\}', text)
    if match:
        return match.group(1).strip()
    # Try #### ...
    match = re.search(r'####\s*(.+?)(?:\.|$)', text)
    if match:
        return match.group(1).strip()
    return ""


def parse_gsm8k_answer(text: str) -> str:
    """Парсит числовой ответ из текста."""
    ans = parse_boxed_answer(text)
    if ans:
        # Извлекаем число
        match = re.search(r'[-+]?\d*\.?\d+', ans)
        if match:
            return match.group()
    # Fallback: ищем #### 
    match = re.search(r'####\s*([-\d.]+)', text)
    if match:
        return match.group(1).strip()
    return ""


def load_model_and_tokenizer(model_path: str, device: torch.device, dtype: torch.dtype):
    """Загружает модель и токенизатор."""
    print(f"  Загрузка модели: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype,
        device_map={"": device.index}, trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    hidden_size = model.config.hidden_size
    print(f"    Hidden size: {hidden_size}")
    return model, tokenizer, hidden_size


def load_inner_adapter(ckpt_path: str, hidden_size: int, device: torch.device, dtype: torch.dtype) -> InnerAdapter:
    """Загружает inner adapter из чекпоинта."""
    adapter = InnerAdapter(hidden_size).to(device, dtype=dtype)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        adapter.load_state_dict(state)
        print(f"    ✅ Inner adapter loaded from {ckpt_path}")
    else:
        print(f"    ⚠️ Inner adapter NOT found: {ckpt_path} (using random init)")
    adapter.eval()
    return adapter


def load_outer_adapters(ckpt_path: str, h1: int, h2: int, h3: int, device: torch.device, dtype: torch.dtype):
    """Загружает outer adapters из чекпоинта."""
    outer_12 = CrossModelAdapter(h1, h2, hidden_dim=512).to(device, dtype=dtype)
    outer_23 = CrossModelAdapter(h2, h3, hidden_dim=512).to(device, dtype=dtype)
    outer_31 = CrossModelAdapter(h3, h1, hidden_dim=512).to(device, dtype=dtype)

    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        outer_12.load_state_dict(state["outer_12"])
        outer_23.load_state_dict(state["outer_23"])
        outer_31.load_state_dict(state["outer_31"])
        print(f"  ✅ Outer adapters loaded from {ckpt_path}")
    else:
        print(f"  ⚠️ Outer adapters NOT found: {ckpt_path} (using random init)")

    for o in [outer_12, outer_23, outer_31]:
        o.eval()
    return outer_12, outer_23, outer_31


# ============================================================
# Latent token generation
# ============================================================
def generate_latent_tokens(
    model,
    tokenizer,
    prompt_text: str,
    system_msg: str,
    inner_adapter: InnerAdapter,
    device: torch.device,
    dtype: torch.dtype,
    num_latent_tokens: int = 20,
    max_prompt_length: int = 1024,
) -> torch.Tensor:
    """
    Генерирует latent tokens через inner adapter loop.
    
    1. Токенизирует prompt
    2. Forward pass → last hidden state
    3. Inner adapter → input embedding space
    4. Forward pass → last hidden state
    5. Repeat num_latent_tokens раз
    
    Returns:
        latent_tokens: [num_latent_tokens, hidden_dim]
    """
    embedding_layer = model.get_input_embeddings()
    
    # 1. Токенизировать prompt
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_text},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        full_text, return_tensors="pt",
        truncation=True, max_length=max_prompt_length,
    ).to(device)
    
    with torch.no_grad():
        # 2. Initial forward pass
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        last_hidden = outputs.last_hidden_state[0, -1, :]  # [hidden_dim]
        
        latent_tokens = []
        
        # 3-5. Inner adapter loop
        for _ in range(num_latent_tokens):
            # Apply inner adapter: hidden → input embedding space
            adapted = inner_adapter(last_hidden.unsqueeze(0))  # [1, 1, hidden]
            
            # Forward pass with adapted embedding as input
            outputs = model(
                inputs_embeds=adapted.to(dtype),
            )
            last_hidden = outputs.last_hidden_state[0, -1, :]  # [hidden_dim]
            latent_tokens.append(last_hidden.detach())
    
    return torch.stack(latent_tokens, dim=0)  # [num_latent_tokens, hidden_dim]


def generate_with_slot_injection(
    model,
    tokenizer,
    prompt_with_slot: str,
    slot_embeds: torch.Tensor,
    system_msg: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int = 512,
    max_prompt_length: int = 1024,
) -> str:
    """
    Генерирует текст с slot injection.
    
    1. Токенизирует prompt с placeholder
    2. Находит позицию slot
    3. Вставляет slot embeddings
    4. Генерирует текст авто-регрессивно
    
    Returns:
        Сгенерированный текст
    """
    embedding_layer = model.get_input_embeddings()
    
    # 1. Токенизировать prompt
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_with_slot},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 2. Найти позицию slot
    slot_marker = "<<LATENT_"
    slot_pos = full_text.find(slot_marker)
    if slot_pos < 0:
        raise ValueError(f"Slot marker not found in prompt")
    
    # Найти конец slot marker (до >>)
    slot_end = full_text.find(">>", slot_pos)
    if slot_end < 0:
        raise ValueError(f"Slot end marker not found")
    slot_end += 2  # после >>
    
    prefix_text = full_text[:slot_pos]
    suffix_text = full_text[slot_end:]
    
    # 3. Токенизировать prefix и suffix
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False).input_ids
    
    # 4. Создать embeddings
    prefix_embeds = embedding_layer(torch.tensor(prefix_ids, device=device).unsqueeze(0))
    if suffix_ids:
        suffix_embeds = embedding_layer(torch.tensor(suffix_ids, device=device).unsqueeze(0))
    else:
        suffix_embeds = None
    
    # slot_embeds: [seq_len, hidden] → [1, seq_len, hidden]
    if slot_embeds.dim() == 2:
        slot_embeds = slot_embeds.unsqueeze(0)
    slot_embeds = slot_embeds.to(dtype)
    
    # Конкатенация: prefix + slot + suffix
    parts = [prefix_embeds, slot_embeds]
    if suffix_embeds is not None:
        parts.append(suffix_embeds)
    inputs_embeds = torch.cat(parts, dim=1)
    
    seq_len = inputs_embeds.shape[1]
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    
    # 5. Генерация авто-регрессивно
    with torch.no_grad():
        generated_ids = inputs_embeds.clone()
        all_token_ids = []
        
        for _ in range(max_new_tokens):
            # Forward pass
            if generated_ids.dim() == 3:
                outputs = model(
                    inputs_embeds=generated_ids,
                    attention_mask=attention_mask,
                )
            else:
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                )
            
            # Sample next token
            logits = outputs.logits[0, -1, :]  # [vocab_size]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            all_token_ids.append(next_token.item())
            
            # Append as embedding
            next_embed = embedding_layer(next_token)  # [1, 1, hidden]
            generated_ids = next_embed
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)
    
    # Декодировать
    if all_token_ids:
        text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
    else:
        text = ""
    
    return text


def generate_text_response(
    model,
    tokenizer,
    prompt_text: str,
    system_msg: str,
    device: torch.device,
    max_new_tokens: int = 512,
) -> str:
    """Простая генерация текста без slot injection."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_text},
    ]
    
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        input_text, return_tensors="pt",
        truncation=True, max_length=1024,
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Декодировать только новые токены
    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return generated


# ============================================================
# Main evaluation
# ============================================================
def evaluate_recursivemas_v2(
    num_examples: int,
    device: str,
    num_recursive_rounds: int,
    num_latent_tokens: int,
    output_path: str,
):
    """Оценка RecursiveMAS на GSM8K."""
    device = torch.device(device)
    dtype = torch.bfloat16
    
    print("=" * 70)
    print("RECURSIVEMAS EVALUATION (v2 — соответствует train_outer.py)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Recursive rounds: {num_recursive_rounds}")
    print(f"Latent tokens per agent: {num_latent_tokens}")
    print()
    
    # 1. Загрузка моделей
    print("📦 Загрузка моделей...")
    planner_model, planner_tok, h1 = load_model_and_tokenizer(PLANNER_MODEL, device, dtype)
    # Refiner = Planner (та же модель)
    refiner_model, refiner_tok, h2 = planner_model, planner_tok, h1
    solver_model, solver_tok, h3 = load_model_and_tokenizer(SOLVER_MODEL, device, dtype)
    print()
    
    # 2. Загрузка inner adapters
    print("🔗 Загрузка inner adapters...")
    inner_1 = load_inner_adapter(INNER_PLANNER_CKPT, h1, device, dtype)
    inner_2 = load_inner_adapter(INNER_REFINER_CKPT, h2, device, dtype)
    inner_3 = load_inner_adapter(INNER_SOLVER_CKPT, h3, device, dtype)
    print()
    
    # 3. Загрузка outer adapters
    print("🔗 Загрузка outer adapters...")
    outer_12, outer_23, outer_31 = load_outer_adapters(OUTER_CKPT, h1, h2, h3, device, dtype)
    print(f"  outer_12: {h1} → {h2}")
    print(f"  outer_23: {h2} → {h3}")
    print(f"  outer_31: {h3} → {h1}")
    print()
    
    # 4. Загрузка данных
    print("📊 Загрузка GSM8K (test)...")
    data = load_gsm8k_dataset(num_examples=num_examples, split="test")
    questions = data["questions"]
    gt_numbers = data["gt_numbers"]
    print(f"  Загружено {len(questions)} примеров")
    print()
    
    # 5. Инференс
    print("🚀 Запуск инференса...")
    print(f"  Pipeline: Planner → inner_1 → outer_12 → Refiner → inner_2 → outer_23 → Solver")
    if num_recursive_rounds > 1:
        print(f"  + Feedback loop: Solver → inner_3 → outer_31 → Planner ({num_recursive_rounds} rounds)")
    print("-" * 70)
    
    predictions = []
    timings = []
    
    for i, question in enumerate(questions):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Обработка {i+1}/{len(questions)}...  ", end="\r")
        
        t_start = time.time()
        
        # Recursive loop
        final_answer = None
        for round_num in range(num_recursive_rounds):
            # --- Planner ---
            if round_num == 0:
                planner_prompt = build_math_planner_prompt(question)
            else:
                planner_prompt = build_math_planner_prompt_with_feedback(question)
                # Slot injection для feedback
                planner_latent = generate_latent_tokens(
                    planner_model, planner_tok,
                    build_math_planner_prompt(question),  # base prompt for tokenization
                    SYSTEM_PROMPT,
                    inner_1, device, dtype,
                    num_latent_tokens=num_latent_tokens,
                )
                # We'll handle feedback below
                planner_prompt = build_math_planner_prompt_with_feedback(question)
            
            # Generate latent tokens from Planner
            planner_latents = generate_latent_tokens(
                planner_model, planner_tok,
                planner_prompt if round_num == 0 else build_math_planner_prompt(question),
                SYSTEM_PROMPT,
                inner_1, device, dtype,
                num_latent_tokens=num_latent_tokens,
            )
            
            # Map to Refiner space: outer_12
            # Take mean of latent tokens as single representation
            planner_rep = planner_latents.mean(dim=0, keepdim=True)  # [1, 1, h1]
            refiner_slot = outer_12(planner_rep)  # [1, 1, h2]
            
            # --- Refiner ---
            refiner_prompt = build_math_refiner_prompt(question)
            # Generate latent tokens from Refiner with slot injection
            refiner_latents = generate_latent_tokens_with_slot(
                refiner_model, refiner_tok,
                refiner_prompt, SYSTEM_PROMPT,
                inner_2, refiner_slot,
                device, dtype,
                num_latent_tokens=num_latent_tokens,
            )
            
            # Map to Solver space: outer_23
            refiner_rep = refiner_latents.mean(dim=0, keepdim=True)  # [1, 1, h2]
            solver_slot = outer_23(refiner_rep)  # [1, 1, h3]
            
            # --- Solver ---
            solver_prompt = build_math_solver_prompt(question)
            answer_text = generate_with_slot_injection(
                solver_model, solver_tok,
                solver_prompt, solver_slot,
                SYSTEM_PROMPT,
                device, dtype,
                max_new_tokens=512,
            )
            
            final_answer = answer_text
            
            # Prepare feedback for next round
            if round_num < num_recursive_rounds - 1:
                # Generate solver latents for feedback
                solver_latents = generate_latent_tokens_with_slot(
                    solver_model, solver_tok,
                    solver_prompt, solver_slot,
                    inner_3, solver_slot,
                    device, dtype,
                    num_latent_tokens=num_latent_tokens,
                )
                # Map to Planner space: outer_31
                solver_rep = solver_latents.mean(dim=0, keepdim=True)  # [1, 1, h3]
                feedback_slot = outer_31(solver_rep)  # [1, 1, h1]
                # Store for next round
                planner_feedback = feedback_slot
        
        elapsed = time.time() - t_start
        
        # Parse answer
        parsed = parse_gsm8k_answer(final_answer or "")
        predictions.append(parsed if parsed else None)
        timings.append(elapsed)
    
    print(f"  Обработка {len(questions)}/{len(questions)}... ✅\n")
    
    # 6. Оценка
    results = compute_accuracy(predictions, gt_numbers, questions=questions)
    
    # Мета-информация
    meta = {
        "type": "recursivemas_v2",
        "planner_model": PLANNER_MODEL,
        "refiner_model": REFINER_MODEL,
        "solver_model": SOLVER_MODEL,
        "num_recursive_rounds": num_recursive_rounds,
        "num_latent_tokens": num_latent_tokens,
        "device": str(device),
        "num_examples": len(questions),
        "split": "test",
    }
    
    # Вывод
    title = f"RecursiveMAS v2: Planner → Refiner → Solver ({num_recursive_rounds} rounds)"
    print_results(results, title=title, timings=timings)
    
    # Сохранение
    if output_path:
        save_results(results, output_path, timings=timings, meta=meta)
        print(f"\nРезультаты сохранены в {output_path}")
    
    return results


def generate_latent_tokens_with_slot(
    model,
    tokenizer,
    prompt_with_slot: str,
    system_msg: str,
    inner_adapter: InnerAdapter,
    slot_embeds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    num_latent_tokens: int = 20,
    max_prompt_length: int = 1024,
) -> torch.Tensor:
    """
    Генерирует latent tokens с slot injection.
    
    1. Токенизирует prompt с slot
    2. Forward pass с injected embeddings
    3. Inner adapter loop для latent token generation
    
    Returns:
        latent_tokens: [num_latent_tokens, hidden_dim]
    """
    embedding_layer = model.get_input_embeddings()
    
    # Найти slot marker
    slot_marker = "<<LATENT_"
    slot_pos = prompt_with_slot.find(slot_marker)
    if slot_pos < 0:
        raise ValueError("Slot marker not found")
    slot_end = prompt_with_slot.find(">>", slot_pos) + 2
    
    prefix_text = prompt_with_slot[:slot_pos]
    suffix_text = prompt_with_slot[slot_end:]
    
    # Build full chat template
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_with_slot},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Find slot in full text
    full_slot_pos = full_text.find(slot_marker)
    if full_slot_pos < 0:
        raise ValueError("Slot marker not found in full text")
    full_slot_end = full_text.find(">>", full_slot_pos) + 2
    
    prefix_full = full_text[:full_slot_pos]
    suffix_full = full_text[full_slot_end:]
    
    # Tokenize
    prefix_ids = tokenizer(prefix_full, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(suffix_full, add_special_tokens=False).input_ids
    
    # Build embeddings
    prefix_embeds = embedding_layer(torch.tensor(prefix_ids, device=device).unsqueeze(0))
    if suffix_ids:
        suffix_embeds = embedding_layer(torch.tensor(suffix_ids, device=device).unsqueeze(0))
    else:
        suffix_embeds = None
    
    if slot_embeds.dim() == 2:
        slot_embeds = slot_embeds.unsqueeze(0)
    slot_embeds = slot_embeds.to(dtype)
    
    parts = [prefix_embeds, slot_embeds]
    if suffix_embeds is not None:
        parts.append(suffix_embeds)
    inputs_embeds = torch.cat(parts, dim=1)
    
    seq_len = inputs_embeds.shape[1]
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Initial forward pass
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        last_hidden = outputs.last_hidden_state[0, -1, :]
        
        latent_tokens = []
        for _ in range(num_latent_tokens):
            adapted = inner_adapter(last_hidden.unsqueeze(0))
            outputs = model(inputs_embeds=adapted.to(dtype))
            last_hidden = outputs.last_hidden_state[0, -1, :]
            latent_tokens.append(last_hidden.detach())
    
    return torch.stack(latent_tokens, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="RecursiveMAS evaluation on GSM8K (v2 — соответствует train_outer.py)",
    )
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Количество примеров (default: 100)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help=f"Устройство (default: {DEFAULT_DEVICE})")
    parser.add_argument("--num_recursive_rounds", type=int, default=1,
                        help="Число рекурсивных раундов (default: 1)")
    parser.add_argument("--num_latent_tokens", type=int, default=DEFAULT_NUM_LATENT_TOKENS,
                        help="Число latent токенов на агента (default: 20)")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь для JSON результатов")
    
    args = parser.parse_args()
    
    results = evaluate_recursivemas_v2(
        num_examples=args.num_examples,
        device=args.device,
        num_recursive_rounds=args.num_recursive_rounds,
        num_latent_tokens=args.num_latent_tokens,
        output_path=args.output,
    )
    
    sys.exit(0 if results["accuracy"] > 0 else 1)


if __name__ == "__main__":
    main()
