#!/usr/bin/env python3
"""
evaluate_single_model.py — Baseline: тест одной модели на GSM8K.

Прототипирует точность отдельных моделей на GSM8K test split.
Поддерживает любые модели через --model.

Примеры:
    python evaluate_single_model.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --num_examples 100
    python evaluate_single_model.py --model Qwen/Qwen3-1.7B --num_examples 100
    python evaluate_single_model.py --model ./models/Qwen2.5-Math-1.5B-Instruct --num_examples 100
"""

import os
import sys
import argparse
import torch

# Добавляем custom_impl в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gsm8k_utils import (
    load_gsm8k_dataset,
    load_model,
    format_gsm8k_prompt,
    generate_answer,
    compute_accuracy,
    print_results,
    save_results,
)

# Пути к моделям по умолчанию
DEFAULT_MODEL = "./models/Qwen2.5-Math-1.5B-Instruct"
DEFAULT_DEVICE = "cuda:2"


def evaluate_single(model_path, num_examples, device, output_path):
    """Оценка одной модели на GSM8K."""

    # Загрузка модели
    model, tokenizer = load_model(model_path, device)

    # Загрузка данных
    data = load_gsm8k_dataset(num_examples=num_examples, split="test")
    questions = data["questions"]
    gt_numbers = data["gt_numbers"]

    # System prompt — математическая специализация
    system_msg = (
        "You are a helpful math assistant. "
        "Solve the following math problem step by step. "
        "Put your final numerical answer after ####, for example: #### 42"
    )

    print(f"\nЗапуск оценки: {model_path}")
    print(f"Примеров: {len(questions)}")
    print(f"Device: {device}")
    print("-" * 70)

    predictions = []
    timings = []

    for i, question in enumerate(questions):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Обработка {i+1}/{len(questions)}...", end="\r")

        messages = format_gsm8k_prompt(question, system_msg=system_msg)

        import time
        t0 = time.time()
        pred = generate_answer(
            model, tokenizer, messages, device,
            max_new_tokens=512,
        )
        elapsed = time.time() - t0
        timings.append(elapsed)

        predictions.append(pred)

    print(f"  Обработка {len(questions)}/{len(questions)}... ✅\n")

    # Оценка
    results = compute_accuracy(predictions, gt_numbers, questions=questions)

    # Мета-информация
    meta = {
        "model": model_path,
        "device": str(device),
        "num_examples": len(questions),
        "split": "test",
        "type": "single_model",
    }

    # Вывод
    print_results(results, title=f"Single Model: {model_path}", timings=timings)

    # Сохранение
    if output_path:
        save_results(results, output_path, timings=timings, meta=meta)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation: single model on GSM8K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Путь или HF имя модели (default: {DEFAULT_MODEL})",
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

    results = evaluate_single(
        model_path=args.model,
        num_examples=args.num_examples,
        device=args.device,
        output_path=args.output,
    )

    # Возврат кода ошибки если accuracy = 0 (что-то пошло не так)
    sys.exit(0 if results["accuracy"] > 0 else 1)


if __name__ == "__main__":
    main()
