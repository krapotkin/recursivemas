#!/usr/bin/env python3
"""
Общие утилиты для загрузки датасетов и парсинга ответов.

Поддерживает: GSM8K, Math500
Унифицированный интерфейс для evaluate_single_model.py, evaluate_text_mas.py, evaluate_recursivemas.py
"""

import os
import re
import json
import time
from typing import Optional, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")


# ============================================================
# Загрузка данных
# ============================================================

DATASET_CONFIGS = {
    "gsm8k": {
        "hf_name": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "answer_format": "gsm8k",  # answer in "text #### number" format
    },
    "math500": {
        "hf_name": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_format": "math500",  # answer is just a number/expression
    },
}


def load_dataset_safe(dataset_name: str, num_examples: int = -1, split: str = "test") -> Dict:
    """Загрузка датасета с поддержкой GSM8K и Math500.

    Returns:
        dict с ключами:
          - questions: list[str]
          - answers: list[str] — оригинальные ответы (raw из датасета)
          - gt_numbers: list[str] — распарсенные числовые ответы
          - dataset_name: str
    """
    cfg = DATASET_CONFIGS.get(dataset_name)
    if cfg is None:
        raise ValueError(f"Неизвестный датасет: {dataset_name}. Доступны: {list(DATASET_CONFIGS.keys())}")

    print(f"Загрузка {dataset_name} ({split})...")
    ds = load_dataset(cfg["hf_name"], cfg["config"])[split]

    questions = ds[cfg["question_field"]]
    answers_raw = ds[cfg["answer_field"]]

    if num_examples > 0:
        questions = questions[:num_examples]
        answers_raw = answers_raw[:num_examples]

    # Парсим GT ответы согласно формату датасета
    gt_numbers = [_parse_gt_answer(a, dataset_name) for a in answers_raw]

    print(f"  Загружено {len(questions)} примеров")
    return {
        "questions": questions,
        "answers": answers_raw,
        "gt_numbers": gt_numbers,
        "dataset_name": dataset_name,
    }


def _parse_gt_answer(answer_text: str, dataset: str) -> str:
    """Извлекает число из ground truth ответа согласно формату датасета."""
    if not answer_text:
        return ""
    if dataset == "gsm8k":
        # Формат: "text #### number"
        m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
        if m:
            return _normalize_number(m.group(1))
        # Fallback: последнее число
        nums = re.findall(r"-?[\d,]+\.?\d*", answer_text)
        if nums:
            return _normalize_number(nums[-1])
        return ""
    else:  # math500
            # Пробуем \boxed{X}
            m = re.search(r"\\boxed\{([^}]+)\}", answer_text)
            if m:
                return _normalize_number(m.group(1))
            # Вероятно, просто число
            return _normalize_number(answer_text.strip())


# ============================================================
# Парсинг и оценка
# ============================================================

def parse_answer(text: str) -> Optional[str]:
    """Парсинг ответа из текста модели.

    Поддерживает форматы:
    - #### X (GSM8K стандарт)
    - \\boxed{X} (LaTeX формат)
    - Fallback: последнее число в тексте

    Returns:
        Нормализованное числовое значение или None.
    """
    if not text:
        return None
    # Формат GSM8K: #### 42
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return normalize_answer(match.group(1))
    # Формат LaTeX: \boxed{42}
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return normalize_answer(match.group(1))
    # Fallback: последнее число
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return normalize_answer(numbers[-1])
    return None


def normalize_answer(answer: str) -> str:
    """Нормализация числового ответа.

    Удаляет запятые, доллары, проценты, извлекает число.
    """
    if not answer:
        return ""
    answer = re.sub(r"[,$%]", "", answer)
    # Remove trailing dot
    answer = answer.rstrip(".")
    match = re.search(r"(-?\d+\.?\d*)", answer.strip())
    if match:
        return match.group(1)
    return answer.strip()


def _normalize_number(s: str) -> str:
    """Внутренняя нормализация числа."""
    s = s.strip().replace(",", "")
    # Remove leading zeros (but keep "0", "0.5")
    if "." in s:
        s = re.sub(r"^0+", "", s) if s != "0" else s
    else:
        s = re.sub(r"^0+", "", s) if s != "0" else "0"
    if s.startswith("."):
        s = "0" + s
    return s


def compute_accuracy(predictions, gt_numbers, questions=None):
    """Вычисление accuracy и сбор детальной статистики.

    Args:
        predictions: список предсказанных текстов
        gt_numbers: список ground truth чисел (уже распарсенных)
        questions: опциональный список вопросов

    Returns:
        dict с accuracy, correct, total, examples
    """
    correct = 0
    total = len(predictions)
    examples = []
    for i, (pred, gt) in enumerate(zip(predictions, gt_numbers)):
        pred_parsed = parse_answer(pred)
        pred_norm = normalize_answer(pred_parsed) if pred_parsed else ""
        gt_norm = normalize_answer(str(gt))
        is_correct = (pred_norm == gt_norm)
        if is_correct:
            correct += 1
        ex = {
            "index": i,
            "prediction": pred,
            "prediction_parsed": pred_parsed,
            "answer_parsed": gt,
            "correct": is_correct,
        }
        if questions:
            ex["question"] = questions[i]
        examples.append(ex)
    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total, "examples": examples}


def print_results(results, title="Результаты", timings=None):
    """Красивая печать результатов в консоль."""
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)
    print(f"  Accuracy: {results['accuracy'] * 100:.1f}% ({results['correct']}/{results['total']})")
    if timings:
        avg_time = sum(timings) / len(timings) if timings else 0
        print(f"  Среднее время на пример: {avg_time:.2f}s")
        print(f"  Общее время: {sum(timings):.1f}s")
    print("=" * 70)

    print("\nПримеры (первые 5):")
    for ex in results["examples"][:5]:
        status = "[OK]" if ex["correct"] else "[FAIL]"
        print(f"  {status} [{ex['index']}] Pred: {ex['prediction_parsed']}, Ans: {ex['answer_parsed']}")

    errors = [ex for ex in results["examples"] if not ex["correct"]]
    if errors:
        print(f"\nПервые 3 ошибки (из {len(errors)}):")
        for ex in errors[:3]:
            q = ex.get("question", "N/A")[:80]
            print(f"  [FAIL] [{ex['index']}] Pred: {ex['prediction_parsed']}, Ans: {ex['answer_parsed']}")

    # Save results to JSON
    save_path = f"eval_results_{int(time.time())}.json"
    save_dict = {
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"],
        "dataset": results.get("dataset", "unknown"),
        "examples": results["examples"][:20],  # keep only first 20 to keep file small
    }
    # Also save full results to timestamped dir
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                               "data", "recursivemas", "results")
    os.makedirs(results_dir, exist_ok=True)
    full_path = os.path.join(results_dir, f"{results.get('dataset', 'unknown')}_{int(time.time())}.json")
    with open(full_path, "w") as f:
        json.dump({
            "accuracy": results["accuracy"],
            "correct": results["correct"],
            "total": results["total"],
            "examples": results["examples"],
        }, f, indent=2)
    print(f"\nРезультаты сохранены в: {full_path}")


# ============================================================
# Загрузка модели
# ============================================================

MODEL_CACHE = {}


def load_model(model_name: str, device: torch.device, dtype: torch.dtype,
               trust_remote_code: bool = True) -> AutoModelForCausalLM:
    """Загрузка модели с кэшированием (чтобы не грузить дважды)."""
    key = f"{model_name}_{device}_{dtype}"
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    print(f"Загрузка модели: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    MODEL_CACHE[key] = model
    return model


def load_tokenizer(model_name: str, trust_remote_code: bool = True) -> AutoTokenizer:
    """Загрузка токенизатора."""
    print(f"Загрузка токенизатора: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def clear_model_cache():
    """Очистка кэша моделей."""
    global MODEL_CACHE
    for key, model in MODEL_CACHE.items():
        del model
    MODEL_CACHE = {}
    torch.cuda.empty_cache()