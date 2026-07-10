#!/usr/bin/env python3
"""
pregenerate_teacher_data.py — ⚠️ УСТАРЕЛ — Артефакт, не используется.

> **НЕ ИСПОЛЬЗУЕТСЯ.** Сохранён как исторический артефакт.
> Подробности: см. `pregenerate_teacher_data_ARCHIVE.md`.
>
> Для обучения используйте: `train_inner.py` и `train_outer.py`
> с датасетом `RecursiveMAS/Sequential-Math` из HuggingFace.

Исторический контекст:
Этот скрипт генерировал plan и refined_plan для GSM8K train split
с помощью Planner и Critic. Результат сохранялся в JSON для использования
при обучении Outer Link.

Мы отказались от этого подхода в пользу оригинального датасета
RecursiveMAS/Sequential-Math, который уже содержит teacher-generated планы.

Пример (устаревший):
    python pregenerate_teacher_data.py --num_samples 5000
"""

import os
import sys
import json
import argparse
import torch
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gsm8k_utils import load_model, generate_answer

# Модели
PLANNER_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_DEVICE = "cuda:2"

# System prompts
PLANNER_SYSTEM = (
    "You are a math problem planner. "
    "Given a math word problem, decompose it into clear, step-by-step instructions "
    "that a solver can follow to arrive at the correct answer. "
    "Focus on the logical flow of the solution. "
    "Do NOT solve the problem — only plan the approach."
)

CRITIC_SYSTEM = (
    "You are a math problem critic. "
    "Review the given solution plan for a math problem. "
    "Identify any gaps, errors, or missing steps. "
    "Produce an improved, more complete plan that addresses all issues. "
    "Your improved plan should be clear and actionable."
)


def run_planner(model, tokenizer, question, device):
    """Planner: генерирует план решения."""
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": question},
    ]
    return generate_answer(model, tokenizer, messages, device, max_new_tokens=256)


def run_critic(model, tokenizer, question, plan, device):
    """Critic: улучшает план."""
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": (
            f"Problem:\n{question}\n\n"
            f"Initial Plan:\n{plan}"
        )},
    ]
    return generate_answer(model, tokenizer, messages, device, max_new_tokens=256)


def pregenerate(num_samples, device, output_path):
    """Генерация teacher data для обучения."""

    # Загрузка модели (Planner = Critic)
    print("Загрузка модели Planner/Critic...")
    model, tokenizer = load_model(PLANNER_MODEL, device)

    # Загрузка GSM8K train
    print("Загрузка GSM8K train...")
    dataset = load_dataset("gsm8k", "main", trust_remote_code=True)["train"]

    if num_samples > len(dataset):
        print(f"⚠️ Запрошено {num_samples}, но доступно {len(dataset)}. Берём все.")
        num_samples = len(dataset)

    questions = dataset["question"][:num_samples]
    answers = dataset["answer"][:num_samples]

    print(f"\nГенерация teacher data для {num_samples} примеров...")
    print(f"Device: {device}")
    print("-" * 70)

    data = []
    for i in range(num_samples):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Обработка {i+1}/{num_samples}...", end="\r")

        question = questions[i]
        answer = answers[i]

        # Stage 1: Planner
        plan = run_planner(model, tokenizer, question, device)

        # Stage 2: Critic
        refined_plan = run_critic(model, tokenizer, question, plan, device)

        data.append({
            "question": question,
            "plan": plan,
            "refined_plan": refined_plan,
            "answer": answer,
        })

    print(f"  Обработка {num_samples}/{num_samples}... ✅\n")

    # Сохранение
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Сохранено {len(data)} примеров в {output_path}")

    # Статистика
    plan_lengths = [len(d["plan"].split()) for d in data]
    refined_lengths = [len(d["refined_plan"].split()) for d in data]
    print(f"\nСтатистика:")
    print(f"  Plan: avg {sum(plan_lengths)/len(plan_lengths):.0f} слов, "
          f"min {min(plan_lengths)}, max {max(plan_lengths)}")
    print(f"  Refined plan: avg {sum(refined_lengths)/len(refined_lengths):.0f} слов, "
          f"min {min(refined_lengths)}, max {max(refined_lengths)}")

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate teacher data (plan + refined_plan) for GSM8K training",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Количество примеров для генерации (default: 5000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Устройство (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь для JSON (default: ~/workspace/tmp/recursivemas/teacher_data/gsm8k_train_{num}.json)",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = f"~/workspace/tmp/recursivemas/teacher_data/gsm8k_train_{args.num_samples}.json"

    args.output = os.path.expanduser(args.output)

    pregenerate(
        num_samples=args.num_samples,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
