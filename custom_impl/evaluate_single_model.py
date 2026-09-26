#!/usr/bin/env python3
"""
evaluate_single_model.py — Baseline: тест одной модели.

Поддерживает: GSM8K, Math500.

Примеры:
    python evaluate_single_model.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --dataset math500
    python evaluate_single_model.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --dataset gsm8k --num_examples 100
"""

import argparse
import os
import sys
import time
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

DEFAULT_MODEL = os.path.expanduser("~/workspace/models/models_llm/Qwen2.5-Math-1.5B-Instruct")
DEFAULT_DEVICE = "cuda:1"


def format_prompt(question: str, tokenizer) -> str:
    """Форматирование промпта для модели через chat_template."""
    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Reason step by step and put your final answer after ####."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_answer(model, tokenizer, prompt: str, device, max_new_tokens=1024) -> str:
    """Генерация ответа модели."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def evaluate_single(model_path, dataset_name, num_examples, device, output_path):
    """Оценка одной модели на указанном датасете."""

    # Загрузка датасета
    data = load_dataset_safe(dataset_name, num_examples=num_examples)
    questions = data["questions"]
    gt_numbers = data["gt_numbers"]
    print(f"Датасет: {dataset_name}, примеров: {len(questions)}")

    # Загрузка модели
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_obj = torch.device(device)
    model = load_model(model_path, device_obj, dtype)
    tokenizer = load_tokenizer(model_path)

    print(f"\nЗапуск оценки: {model_path}")
    print(f"Device: {device}")
    print("-" * 70)

    predictions = []
    timings = []

    for i, question in enumerate(questions):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Обработка {i+1}/{len(questions)}...", end="\r")

        prompt = format_prompt(question, tokenizer)

        t0 = time.time()
        pred = generate_answer(model, tokenizer, prompt, device_obj, max_new_tokens=1024)
        elapsed = time.time() - t0
        timings.append(elapsed)
        predictions.append(pred)

    print(f"  Обработка {len(questions)}/{len(questions)}... Done\n")

    # Оценка
    results = compute_accuracy(predictions, gt_numbers, questions=questions)
    results["dataset"] = dataset_name

    # Вывод
    print_results(results, title=f"Single Model: {model_path}", timings=timings)

    # Сохранение
    if output_path:
        save_dict = {
            "model": model_path,
            "dataset": dataset_name,
            "accuracy": results["accuracy"],
            "correct": results["correct"],
            "total": results["total"],
            "device": device,
        }
        with open(output_path, "w") as f:
            import json
            json.dump(save_dict, f, indent=2)
        print(f"Результаты сохранены в: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation: single model on GSM8K / Math500",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Путь или HF имя модели (default: {DEFAULT_MODEL})")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math500"],
                        help="Датасет для оценки (default: gsm8k)")
    parser.add_argument("--num_examples", type=int, default=-1,
                        help="Количество примеров для тестирования (default: -1 = все)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help=f"Устройство (default: {DEFAULT_DEVICE})")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь для JSON результатов (default: не сохранять)")

    args = parser.parse_args()

    results = evaluate_single(
        model_path=args.model,
        dataset_name=args.dataset,
        num_examples=args.num_examples,
        device=args.device,
        output_path=args.output,
    )

    sys.exit(0 if results["accuracy"] > 0 else 1)


if __name__ == "__main__":
    main()