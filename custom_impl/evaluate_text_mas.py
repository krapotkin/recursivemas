#!/usr/bin/env python3
"""
evaluate_text_mas.py — Тест текстового MAS (Planner → Critic → Solver через текст).

Цепочка:
  1. Planner (Qwen3-1.7B) генерирует план решения
  2. Critic (Qwen3-1.7B) улучшает план
  3. Solver (Qwen2.5-Math-1.5B-Instruct) решает задачу с улучшенным планом

Коммуникация: текст (стандартный chat template).
Это baseline для сравнения с латентной коммуникацией (RecursiveMAS).

Пример:
    python evaluate_text_mas.py --num_examples 100
"""

import os
import sys
import time
import argparse
import torch

# Добавляем custom_impl в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gsm8k_utils import (
    load_gsm8k_dataset,
    load_model,
    generate_answer,
    compute_accuracy,
    print_results,
    save_results,
)

# Модели
PLANNER_MODEL = "Qwen/Qwen3-1.7B"          # Planner + Critic (одна модель)
SOLVER_MODEL = "/home/hermes/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct"
DEFAULT_DEVICE = "cuda:2"

# System prompts для каждой роли
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

SOLVER_SYSTEM = (
    "You are a math problem solver. "
    "Solve the math problem step by step, following the provided plan. "
    "Show your work clearly. "
    "Put your final numerical answer after ####, for example: #### 42"
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


def run_solver(solver_model, solver_tokenizer, question, refined_plan, device):
    """Solver: решает задачу с планом."""
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM},
        {"role": "user", "content": (
            f"Problem:\n{question}\n\n"
            f"Solution Plan:\n{refined_plan}"
        )},
    ]
    return generate_answer(solver_model, solver_tokenizer, messages, device, max_new_tokens=512)


def evaluate_text_mas(num_examples, device, output_path):
    """Оценка текстового MAS на GSM8K."""

    # Загрузка моделей (Planner и Critic — одна модель)
    print("Загрузка моделей...")
    planner_model, planner_tokenizer = load_model(PLANNER_MODEL, device)
    solver_model, solver_tokenizer = load_model(SOLVER_MODEL, device)

    # Загрузка данных
    data = load_gsm8k_dataset(num_examples=num_examples, split="test")
    questions = data["questions"]
    gt_numbers = data["gt_numbers"]

    print(f"\nЗапуск текстового MAS: Planner → Critic → Solver")
    print(f"Planner/Critic: {PLANNER_MODEL}")
    print(f"Solver: {SOLVER_MODEL}")
    print(f"Примеров: {len(questions)}")
    print(f"Device: {device}")
    print("-" * 70)

    predictions = []
    timings = []
    stage_timings = {"planner": [], "critic": [], "solver": []}

    for i, question in enumerate(questions):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Обработка {i+1}/{len(questions)}...", end="\r")

        t_start = time.time()

        # Stage 1: Planner
        t0 = time.time()
        plan = run_planner(planner_model, planner_tokenizer, question, device)
        stage_timings["planner"].append(time.time() - t0)

        # Stage 2: Critic
        t0 = time.time()
        refined_plan = run_critic(planner_model, planner_tokenizer, question, plan, device)
        stage_timings["critic"].append(time.time() - t0)

        # Stage 3: Solver
        t0 = time.time()
        pred = run_solver(solver_model, solver_tokenizer, question, refined_plan, device)
        stage_timings["solver"].append(time.time() - t0)

        elapsed = time.time() - t_start
        timings.append(elapsed)
        predictions.append(pred)

    print(f"  Обработка {len(questions)}/{len(questions)}... Done\n")

    # Оценка
    results = compute_accuracy(predictions, gt_numbers, questions=questions)

    # Мета-информация
    meta = {
        "type": "text_mas",
        "planner_model": PLANNER_MODEL,
        "critic_model": PLANNER_MODEL,
        "solver_model": SOLVER_MODEL,
        "device": str(device),
        "num_examples": len(questions),
        "split": "test",
        "stage_avg_times": {
            "planner": sum(stage_timings["planner"]) / len(stage_timings["planner"]),
            "critic": sum(stage_timings["critic"]) / len(stage_timings["critic"]),
            "solver": sum(stage_timings["solver"]) / len(stage_timings["solver"]),
        },
    }

    # Вывод
    title = f"Text MAS: Planner({PLANNER_MODEL.split('/')[-1]}) → Critic → Solver({SOLVER_MODEL.split('/')[-1]})"
    print_results(results, title=title, timings=timings)

    # Timing по стадиям
    print("\nСреднее время по стадиям:")
    for stage, times in stage_timings.items():
        avg = sum(times) / len(times) if times else 0
        print(f"  {stage}: {avg:.2f}s")

    # Сохранение
    if output_path:
        save_results(results, output_path, timings=timings, meta=meta)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Text-based MAS evaluation on GSM8K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Количество примеров для тестирования (default: 100)",
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
        help="Путь для JSON результатов (default: не сохранять)",
    )

    args = parser.parse_args()

    results = evaluate_text_mas(
        num_examples=args.num_examples,
        device=args.device,
        output_path=args.output,
    )

    sys.exit(0 if results["accuracy"] > 0 else 1)


if __name__ == "__main__":
    main()
