#!/usr/bin/env python3
"""
evaluate_text_mas.py — Тест текстового MAS на GSM8K / Math500.

Цепочка:
  1. Planner (Qwen3-1.7B) генерирует план решения
  2. Critic (Qwen3-1.7B) улучшает план
  3. Solver (Qwen2.5-Math-1.5B-Instruct) решает задачу с улучшенным планом

Пример:
    python evaluate_text_mas.py --dataset math500 --num_examples 100
    python evaluate_text_mas.py --dataset gsm8k
"""

import os
import sys
import time
import argparse
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_utils import (
    load_dataset_safe,
    load_model,
    load_tokenizer,
    compute_accuracy,
    print_results,
    clear_model_cache,
)

# Модели
PLANNER_MODEL = "Qwen/Qwen3-1.7B"
SOLVER_MODEL = os.path.expanduser("~/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct")
DEFAULT_DEVICE = "cuda:1"


def generate_text(model, tokenizer, messages, device, max_new_tokens=256) -> str:
    """Генерация текста через chat template."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def run_planner(model, tokenizer, question, device):
    messages = [
        {"role": "system", "content": "You are a math problem planner. Decompose the problem into step-by-step instructions. Do NOT solve — only plan."},
        {"role": "user", "content": question},
    ]
    return generate_text(model, tokenizer, messages, device, max_new_tokens=256)


def run_critic(model, tokenizer, question, plan, device):
    messages = [
        {"role": "system", "content": "You are a math problem critic. Review and improve the plan. Identify gaps and errors."},
        {"role": "user", "content": f"Problem:\n{question}\n\nInitial Plan:\n{plan}"},
    ]
    return generate_text(model, tokenizer, messages, device, max_new_tokens=256)


def run_solver(model, tokenizer, question, refined_plan, device):
    messages = [
        {"role": "system", "content": "You are a math problem solver. Solve the problem following the plan. Put final answer after #### or in \\boxed{}."},
        {"role": "user", "content": f"Problem:\n{question}\n\nSolution Plan:\n{refined_plan}"},
    ]
    return generate_text(model, tokenizer, messages, device, max_new_tokens=512)


def evaluate_text_mas(dataset_name, num_examples, device, output_path):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_obj = torch.device(device)

    # Load models
    print("Loading models...")
    planner_model, planner_tok = load_model(PLANNER_MODEL, device_obj, dtype), load_tokenizer(PLANNER_MODEL)
    solver_model, solver_tok = load_model(SOLVER_MODEL, device_obj, dtype), load_tokenizer(SOLVER_MODEL)

    # Load dataset
    data = load_dataset_safe(dataset_name, num_examples=num_examples)
    questions = data["questions"]
    gt_numbers = data["gt_numbers"]
    print(f"Dataset: {dataset_name}, examples: {len(questions)}")

    print(f"\nText MAS: Planner → Critic → Solver")
    print(f"Planner/Critic: {PLANNER_MODEL}")
    print(f"Solver: {SOLVER_MODEL}")
    print(f"Device: {device}")
    print("-" * 70)

    predictions = []
    timings = []
    stage_timings = {"planner": [], "critic": [], "solver": []}

    for i, question in enumerate(questions):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(questions)}...", end="\r")

        t_start = time.time()

        t0 = time.time()
        plan = run_planner(planner_model, planner_tok, question, device_obj)
        stage_timings["planner"].append(time.time() - t0)

        t0 = time.time()
        refined_plan = run_critic(planner_model, planner_tok, question, plan, device_obj)
        stage_timings["critic"].append(time.time() - t0)

        t0 = time.time()
        pred = run_solver(solver_model, solver_tok, question, refined_plan, device_obj)
        stage_timings["solver"].append(time.time() - t0)

        timings.append(time.time() - t_start)
        predictions.append(pred)

    print(f"  Processing {len(questions)}/{len(questions)}... Done\n")

    results = compute_accuracy(predictions, gt_numbers, questions=questions)
    results["dataset"] = dataset_name

    title = f"Text MAS ({dataset_name}): Planner → Critic → Solver"
    print_results(results, title=title, timings=timings)

    print("\nStage avg times:")
    for stage, times in stage_timings.items():
        avg = sum(times) / len(times) if times else 0
        print(f"  {stage}: {avg:.2f}s")

    if output_path:
        with open(output_path, "w") as f:
            json.dump({
                "dataset": dataset_name,
                "accuracy": results["accuracy"],
                "correct": results["correct"],
                "total": results["total"],
                "stage_avg_times": {k: sum(v)/len(v) if v else 0 for k, v in stage_timings.items()},
            }, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Text MAS evaluation on GSM8K / Math500")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()
    results = evaluate_text_mas(
        dataset_name=args.dataset,
        num_examples=args.num_examples,
        device=args.device,
        output_path=args.output,
    )
    sys.exit(0 if results["accuracy"] > 0 else 1)


if __name__ == "__main__":
    main()