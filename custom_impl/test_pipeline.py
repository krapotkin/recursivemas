#!/usr/bin/env python3
"""
Быстрый тест RecursiveMAS pipeline на 1 примере GSM8K.
Проверяет:
1. Загрузку моделей
2. Генерацию Plan (Planner)
3. Генерацию Refined Plan (Critic)
4. Forward pass с teacher forcing
"""

import os
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

import torch
import sys
sys.path.insert(0, "/home/hermes/workspace/projects/recursivemas/custom_impl")

from outer_link_training_v2 import (
    RecursiveMASPipeline,
    CONFIG,
    build_stage_with_slot,
    inject_slot_embeds,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_chat_template():
    """Тест chat template"""
    print("=" * 60)
    print("ТЕСТ 1: Chat Template")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"Chat template:\n{full_text}")
    print()

def test_slot_injection():
    """Тест slot injection"""
    print("=" * 60)
    print("ТЕСТ 2: Slot Injection")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    
    pack = build_stage_with_slot(
        tokenizer=tokenizer,
        user_prompt="Question: What is 2+2?",
        assistant_text="The answer is 4.",
        max_length=128,
    )
    
    print(f"Input IDs: {pack['input_ids']}")
    print(f"Assistant mask: {pack['assistant_mask']}")
    print(f"Labels: {pack['labels']}")
    print(f"Full text:\n{pack['full_text']}")
    print()

def test_generation():
    """Тест генерации Planner и Critic"""
    print("=" * 60)
    print("ТЕСТ 3: Генерация (Planner → Critic)")
    print("=" * 60)
    
    device = "cuda:2"
    
    # Загрузка Planner
    print("Загрузка Planner (Qwen3-1.7B)...")
    planner_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    planner_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map=device,
    )
    planner_model.eval()
    
    # Загрузка Solver
    print("Загрузка Solver (Qwen2.5-Math-1.5B-Instruct)...")
    solver_tokenizer = AutoTokenizer.from_pretrained("./models/Qwen2.5-Math-1.5B-Instruct")
    solver_model = AutoModelForCausalLM.from_pretrained(
        "./models/Qwen2.5-Math-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map=device,
    )
    solver_model.eval()
    
    # Тестовый вопрос
    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    answer = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
    
    # Генерация плана
    print("\nГенерация плана (Planner)...")
    messages = [
        {"role": "system", "content": "You are a planner. Decompose the problem into steps."},
        {"role": "user", "content": question},
    ]
    
    input_text = planner_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = planner_tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = planner_model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
        )
    
    plan = planner_tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    print(f"Plan:\n{plan[:300]}")
    
    # Генерация refined plan
    print("\nГенерация refined plan (Critic)...")
    messages = [
        {"role": "system", "content": "You are a critic. Review and improve the plan."},
        {"role": "user", "content": f"Question: {question}\n\nInitial Plan:\n{plan}"},
    ]
    
    input_text = planner_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = planner_tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = planner_model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
        )
    
    refined_plan = planner_tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    print(f"Refined Plan:\n{refined_plan[:300]}")
    
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Forward pass с teacher forcing")
    print("=" * 60)
    
    # Forward pass через Planner с teacher forcing
    planner_pack = build_stage_with_slot(
        tokenizer=planner_tokenizer,
        user_prompt=question,
        assistant_text=plan,
        max_length=256,
    )
    
    planner_input_ids = planner_pack["input_ids"].unsqueeze(0).to(device)
    planner_attn_mask = planner_pack["attention_mask"].unsqueeze(0).to(device)
    planner_assist_mask = planner_pack["assistant_mask"].to(device)
    
    with torch.no_grad():
        planner_outputs = planner_model(
            input_ids=planner_input_ids,
            attention_mask=planner_attn_mask,
            output_hidden_states=True,
        )
    
    planner_hidden = planner_outputs.hidden_states[-1][0][planner_assist_mask]
    print(f"Planner hidden shape: {planner_hidden.shape}")
    
    # Forward pass через Solver с teacher forcing
    solver_pack = build_stage_with_slot(
        tokenizer=solver_tokenizer,
        user_prompt=f"Question: {question}\n\nPlan:\n{plan}",
        assistant_text=answer,
        max_length=256,
    )
    
    solver_input_ids = solver_pack["input_ids"].unsqueeze(0).to(device)
    solver_attn_mask = solver_pack["attention_mask"].unsqueeze(0).to(device)
    solver_labels = solver_pack["labels"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        solver_outputs = solver_model(
            input_ids=solver_input_ids,
            attention_mask=solver_attn_mask,
            labels=solver_labels,
        )
    
    print(f"Solver loss (baseline): {solver_outputs.loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ! 🎉")
    print("=" * 60)

if __name__ == "__main__":
    test_chat_template()
    test_slot_injection()
    test_generation()
