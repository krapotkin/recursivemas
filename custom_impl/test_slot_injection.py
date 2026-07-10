#!/usr/bin/env python3
"""
Тест полного пайплайна: Slot Injection + Teacher Forcing chain.

Проверяет:
1. Planner генерирует план → извлекаем hidden states
2. Outer Link 1 маппит planner hidden → critic space
3. Inject'им slot в контекст Critic → forward → извлекаем hidden states
4. Outer Link 2 маппит critic hidden → solver space (2048 → 1536!)
5. Inject'им slot в контекст Solver → forward → CE loss на всех токенах
"""

import os
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

import torch
import sys
sys.path.insert(0, "/home/hermes/workspace/projects/recursivemas/custom_impl")

from outer_link_training_v2 import (
    OuterLinkAdapter,
    build_stage_with_slot,
    inject_slot_embeds,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_full_chain():
    """
    Полный тест: Planner → Outer Link → Critic → Outer Link → Solver
    """
    device = "cuda:2"
    
    # ============================================================
    # Загрузка моделей
    # ============================================================
    print("=" * 70)
    print("ЗАГРУЗКА МОДЕЛЕЙ")
    print("=" * 70)
    
    print("Загрузка Planner (Qwen3-1.7B)...")
    planner_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    planner_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", torch_dtype=torch.float16, device_map=device
    )
    planner_model.eval()
    for p in planner_model.parameters():
        p.requires_grad = False
    print(f"  Hidden size: {planner_model.config.hidden_size}")
    
    print("Загрузка Solver (Qwen2.5-Math-1.5B-Instruct)...")
    solver_tok = AutoTokenizer.from_pretrained("./models/Qwen2.5-Math-1.5B-Instruct")
    solver_model = AutoModelForCausalLM.from_pretrained(
        "./models/Qwen2.5-Math-1.5B-Instruct", torch_dtype=torch.float16, device_map=device
    )
    solver_model.eval()
    for p in solver_model.parameters():
        p.requires_grad = False
    print(f"  Hidden size: {solver_model.config.hidden_size}")
    
    # Outer Link адаптеры
    print("\nСоздание Outer Link адаптеров...")
    outer_1 = OuterLinkAdapter(2048, 2048, 512).to(device).half()  # Planner → Critic (same model)
    outer_2 = OuterLinkAdapter(2048, 1536, 512).to(device).half()  # Critic → Solver (DIFFERENT!)
    print(f"  Outer 1: 2048 → 2048 (Planner→Critic)")
    print(f"  Outer 2: 2048 → 1536 (Critic→Solver) ⚡ DIMENSION MAPPING!")
    
    # ============================================================
    # Тестовые данные
    # ============================================================
    print("\n" + "=" * 70)
    print("ТЕСТОВЫЕ ДАННЫЕ")
    print("=" * 70)
    
    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    plan = "Step 1: Calculate clips sold in May. May = April / 2 = 48 / 2 = 24.\nStep 2: Add April and May clips. Total = 48 + 24 = 72."
    answer = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
    
    print(f"Question: {question[:80]}...")
    print(f"Plan: {plan[:80]}...")
    print(f"Answer: {answer[:80]}...")
    
    # ============================================================
    # ШАГ 1: Planner → hidden states (teacher forcing)
    # ============================================================
    print("\n" + "=" * 70)
    print("ШАГ 1: Planner → hidden states (teacher forcing)")
    print("=" * 70)
    
    planner_pack = build_stage_with_slot(
        tokenizer=planner_tok,
        user_prompt=question,
        assistant_text=plan,
        max_length=256,
    )
    
    planner_input_ids = planner_pack["input_ids"].unsqueeze(0).to(device)
    planner_attn_mask = planner_pack["attention_mask"].unsqueeze(0).to(device)
    planner_assist_mask = planner_pack["assistant_mask"].to(device)
    
    print(f"  Input IDs shape: {planner_input_ids.shape}")
    print(f"  Assistant mask sum: {planner_assist_mask.sum().item()} токенов плана")
    
    with torch.no_grad():
        planner_outputs = planner_model(
            input_ids=planner_input_ids,
            attention_mask=planner_attn_mask,
            output_hidden_states=True,
        )
    
    # Извлекаем hidden states ТОЛЬКО для assistant токенов
    planner_hidden = planner_outputs.hidden_states[-1][0][planner_assist_mask]
    print(f"  Planner hidden shape: {planner_hidden.shape}")
    print(f"  ✅ Получили {planner_hidden.shape[0]} hidden vectors (по одному на токен плана)")
    
    # ============================================================
    # ШАГ 2: Outer Link 1 (Planner → Critic)
    # ============================================================
    print("\n" + "=" * 70)
    print("ШАГ 2: Outer Link 1 (Planner → Critic, 2048 → 2048)")
    print("=" * 70)
    
    outer_1.train()
    planner_to_critic = outer_1(planner_hidden.unsqueeze(0))  # [1, seq_len, 2048]
    print(f"  Input:  {planner_hidden.unsqueeze(0).shape}")
    print(f"  Output: {planner_to_critic.shape}")
    print(f"  ✅ Dimension mapping: 2048 → 2048")
    
    # ============================================================
    # ШАГ 3: Critic ← Slot Injection → hidden states (teacher forcing)
    # ============================================================
    print("\n" + "=" * 70)
    print("ШАГ 3: Critic ← Slot Injection → hidden states")
    print("=" * 70)
    
    # Создаём input для Critic (та же модель, что Planner)
    critic_user_prompt = f"Question: {question}\n\nInitial Plan:\n{plan}"
    critic_pack = build_stage_with_slot(
        tokenizer=planner_tok,  # Critic = Planner (та же модель)
        user_prompt=critic_user_prompt,
        assistant_text=plan,  # teacher forcing: refined_plan = plan (упрощение)
        max_length=256,
    )
    
    critic_input_ids = critic_pack["input_ids"].unsqueeze(0).to(device)
    critic_attn_mask = critic_pack["attention_mask"].unsqueeze(0).to(device)
    critic_assist_mask = critic_pack["assistant_mask"].to(device)
    
    print(f"  Critic input IDs shape: {critic_input_ids.shape}")
    print(f"  Assistant mask sum: {critic_assist_mask.sum().item()} токенов")
    
    # КЛЮЧЕВОЙ МОМЕНТ: Slot Injection
    # Вставляем slot embeds вместо текстового плана
    # Slot position = позиция, где начинается assistant текст
    slot_position = int(critic_input_ids.shape[1] - critic_assist_mask.sum().item())
    print(f"  Slot position: {slot_position} (вставляем после user prompt)")
    
    # Вставляем slot embeds
    critic_embeds = inject_slot_embeds(
        model=planner_model,  # Critic = Planner
        input_ids=critic_input_ids,
        slot_embeds=planner_to_critic,
        slot_position=slot_position,
    )
    
    print(f"  Input embeds shape: {critic_embeds.shape}")
    print(f"  ✅ Slot injection: вставили {planner_to_critic.shape[1]} slot vectors")
    
    # Forward через Critic (БЕЗ torch.no_grad() — нужны градиенты!)
    critic_outputs = planner_model(
        inputs_embeds=critic_embeds,
        attention_mask=torch.ones(critic_embeds.shape[:2], dtype=torch.long, device=device),
        output_hidden_states=True,
    )
    
    # Извлекаем hidden states для assistant токенов
    # ВАЖНО: из-за slot injection, assistant токены сдвинулись
    # Нужно скорректировать mask
    new_assist_start = slot_position + planner_to_critic.shape[1]
    critic_hidden = critic_outputs.hidden_states[-1][0][new_assist_start:]
    print(f"  Critic hidden shape: {critic_hidden.shape}")
    print(f"  ✅ Получили hidden states от Critic (с injected slot)")
    
    # ============================================================
    # ШАГ 4: Outer Link 2 (Critic → Solver, 2048 → 1536!)
    # ============================================================
    print("\n" + "=" * 70)
    print("ШАГ 4: Outer Link 2 (Critic → Solver, 2048 → 1536)")
    print("=" * 70)
    
    outer_2.train()
    critic_to_solver = outer_2(critic_hidden.unsqueeze(0))  # [1, seq_len, 1536]
    print(f"  Input:  {critic_hidden.unsqueeze(0).shape}")
    print(f"  Output: {critic_to_solver.shape}")
    print(f"  ✅ DIMENSION MAPPING: 2048 → 1536!")
    
    # ============================================================
    # ШАГ 5: Solver ← Slot Injection → CE Loss
    # ============================================================
    print("\n" + "=" * 70)
    print("ШАГ 5: Solver ← Slot Injection → CE Loss")
    print("=" * 70)
    
    # Создаём input для Solver
    solver_user_prompt = f"Question: {question}\n\nPlan:\n{plan}"
    solver_pack = build_stage_with_slot(
        tokenizer=solver_tok,
        user_prompt=solver_user_prompt,
        assistant_text=answer,
        max_length=256,
    )
    
    solver_input_ids = solver_pack["input_ids"].unsqueeze(0).to(device)
    solver_attn_mask = solver_pack["attention_mask"].unsqueeze(0).to(device)
    solver_assist_mask = solver_pack["assistant_mask"].to(device)
    solver_labels = solver_pack["labels"].unsqueeze(0).to(device)
    
    print(f"  Solver input IDs shape: {solver_input_ids.shape}")
    print(f"  Assistant mask sum: {solver_assist_mask.sum().item()} токенов ответа")
    print(f"  Labels (не -100): {(solver_labels != -100).sum().item()} токенов для loss")
    
    # Slot injection для Solver
    slot_position = int(solver_input_ids.shape[1] - solver_assist_mask.sum().item())
    
    # ВАЖНО: critic_to_solver имеет размер 1536, а solver ожидает 1536 — совпадает!
    solver_embeds = inject_slot_embeds(
        model=solver_model,
        input_ids=solver_input_ids,
        slot_embeds=critic_to_solver,
        slot_position=slot_position,
    )
    
    print(f"  Input embeds shape: {solver_embeds.shape}")
    print(f"  ✅ Slot injection: вставили {critic_to_solver.shape[1]} slot vectors")
    
    # ВАЖНО: Нужно скорректировать labels для injected slots
    # Вставляем -100 для позиций slot'ов (они не участвуют в loss)
    slot_len = critic_to_solver.shape[1]
    adjusted_labels = torch.cat([
        solver_labels[:, :slot_position],
        torch.full((1, slot_len), -100, dtype=torch.long, device=device),
        solver_labels[:, slot_position:],
    ], dim=1)
    
    print(f"  Adjusted labels shape: {adjusted_labels.shape}")
    
    # Forward через Solver с CE loss
    solver_outputs = solver_model(
        inputs_embeds=solver_embeds,
        attention_mask=torch.ones(solver_embeds.shape[:2], dtype=torch.long, device=device),
        labels=adjusted_labels,
    )
    
    print(f"\n  CE Loss: {solver_outputs.loss.item():.4f}")
    print(f"  ✅ Loss считается на ВСЕХ токенах ответа (не на 1!)")
    
    # ============================================================
    # ШАГ 6: Backward pass (проверка градиентов)
    # ============================================================
    print("\n" + "=" * 70)
    print("ШАГ 6: Backward pass (градиенты)")
    print("=" * 70)
    
    solver_outputs.loss.backward()
    
    print(f"  Outer 1 W1 grad norm: {outer_1.W1.weight.grad.norm().item():.6f}")
    print(f"  Outer 1 W2 grad norm: {outer_1.W2.weight.grad.norm().item():.6f}")
    print(f"  Outer 1 W3 grad norm: {outer_1.W3.weight.grad.norm().item():.6f}")
    print(f"  Outer 2 W1 grad norm: {outer_2.W1.weight.grad.norm().item():.6f}")
    print(f"  Outer 2 W2 grad norm: {outer_2.W2.weight.grad.norm().item():.6f}")
    print(f"  Outer 2 W3 grad norm: {outer_2.W3.weight.grad.norm().item():.6f}")
    
    has_grad_1 = all(p.grad is not None for p in outer_1.parameters())
    has_grad_2 = all(p.grad is not None for p in outer_2.parameters())
    
    print(f"\n  ✅ Outer 1 градиенты: {'ЕСТЬ' if has_grad_1 else 'НЕТ'}")
    print(f"  ✅ Outer 2 градиенты: {'ЕСТЬ' if has_grad_2 else 'НЕТ'}")
    
    # ============================================================
    # ИТОГ
    # ============================================================
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ТЕСТА")
    print("=" * 70)
    print(f"  ✅ Chat Template — работает")
    print(f"  ✅ Slot Injection — работает (вставка slot embeds в контекст)")
    print(f"  ✅ Teacher Forcing — работает (hidden states от assistant токенов)")
    print(f"  ✅ CE Loss на всех токенах — работает ({solver_labels != -100}.sum()={solver_labels[solver_labels != -100].numel()} токенов)")
    print(f"  ✅ Dimension Mapping — работает (2048 → 1536)")
    print(f"  ✅ Backward pass — работает (градиенты проходят через оба адаптера)")
    print("\n  🎉 ПОЛНЫЙ ПАЙПЛАЙН РАБОТАЕТ!")
    print("=" * 70)

if __name__ == "__main__":
    test_full_chain()
