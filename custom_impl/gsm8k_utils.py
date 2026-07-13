#!/usr/bin/env python3
"""
Общие утилиты для тестирования на GSM8K.
Используется в: evaluate_single_model.py, evaluate_text_mas.py, evaluate_recursivemas.py
"""

import os
import re
import json
import time
from typing import Optional
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")


# ============================================================
# Загрузка данных
# ============================================================

def load_gsm8k_dataset(num_examples=None, split="test"):
    """Загрузка GSM8K датасета.

    Returns:
        dict с ключами 'questions' и 'answers' (ground truth numbers).
    """
    print(f"Загрузка GSM8K ({split})...")
    dataset = load_dataset("openai/gsm8k", "main")[split]
    if num_examples:
        questions = dataset["question"][:num_examples]
        answers = dataset["answer"][:num_examples]
    else:
        questions = dataset["question"]
        answers = dataset["answer"]
    # Парсим ground truth ответы (формат: "... #### 42")
    gt_numbers = [parse_answer(a) for a in answers]
    print(f"  Загружено {len(questions)} примеров")
    return {"questions": questions, "answers": answers, "gt_numbers": gt_numbers}


# ============================================================
# Парсинг и оценка
# ============================================================

def parse_answer(text: str) -> Optional[str]:
    """Парсинг ответа из текста.

    Поддерживает форматы:
    - #### X (GSM8K стандарт)
    - \boxed{X} (LaTeX формат)
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
    match = re.search(r"\\boxed\{(.+?)\}", text)
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
    # Remove commas, dollars, percent signs
    answer = re.sub(r"[,$%]", "", answer)
    # Extract number
    match = re.search(r"(-?\d+\.?\d*)", answer.strip())
    if match:
        return match.group(1)
    return answer.strip()


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
        # Нормализуем оба значения для сравнения
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
            print(f"     Q: {q}...")


def save_results(results, filepath, timings=None, meta=None):
    """Сохранение результатов в JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    output = {
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"],
        "examples": results["examples"],
    }
    if timings:
        output["timings"] = {
            "per_example": timings,
            "avg": sum(timings) / len(timings) if timings else 0,
            "total": sum(timings),
        }
    if meta:
        output["meta"] = meta
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {filepath}")


# ============================================================
# Модели
# ============================================================

def load_model(model_path, device="cuda:2"):
    """Загрузка модели и токенизатора.

    Args:
        model_path: путь или HF имя модели
        device: строка устройства ('cuda:2', 'cpu', ...)

    Returns:
        (model, tokenizer)
    """
    print(f"Загрузка модели: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Hidden size: {model.config.hidden_size}")
    return model, tokenizer


# ============================================================
# Генерация
# ============================================================

def format_gsm8k_prompt(question, system_msg="You are a helpful math assistant. "
                                             "Reason step by step and put your final answer after ####."):
    """Форматирует prompt для GSM8K задачи.

    Returns:
        list[dict] — сообщения для apply_chat_template.
    """
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ]


def generate_answer(model, tokenizer, messages, device, max_new_tokens=512,
                    temperature=None, do_sample=None):
    """Генерация ответа с помощью модели.

    Args:
        model: AutoModelForCausalLM
        tokenizer: AutoTokenizer
        messages: list[dict] — chat messages
        device: устройство
        max_new_tokens: макс. новых токенов
        temperature: температура (None = greedy)
        do_sample: sampling (None = greedy по умолчанию)

    Returns:
        str — сгенерированный текст.
    """
    if do_sample is None:
        do_sample = temperature is not None

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
        "do_sample": do_sample,
    }
    if temperature is not None:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return generated


def generate_with_hidden(model, tokenizer, messages, device, max_new_tokens=512):
    """Генерация ответа + извлечение hidden states assistant токенов.

    Использует teacher forcing: forward pass с input_ids,
    извлекает last-layer hidden states.

    Args:
        model: AutoModelForCausalLM
        tokenizer: AutoTokenizer
        messages: list[dict] — chat messages (должен содержать assistant ответ)
        device: устройство
        max_new_tokens: макс. новых токенов (для генерации, если нет assistant)

    Returns:
        dict с 'text' (сгенерированный текст) и 'hidden' (hidden states assistant токенов).
    """
    # Генерируем ответ
    text = generate_answer(model, tokenizer, messages, device, max_new_tokens=max_new_tokens)

    # Для извлечения hidden states делаем forward pass с полным контекстом
    full_messages = messages + [{"role": "assistant", "content": text}]
    input_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        )

    # Находим границу assistant токенов
    # Токенизируем только assistant текст
    assistant_tokens = tokenizer(text, add_special_tokens=False).input_ids
    assistant_start = inputs.input_ids.shape[1] - len(assistant_tokens)

    # Извлекаем hidden states assistant токенов
    hidden = outputs.hidden_states[-1][0, assistant_start:, :]  # [seq_len, hidden]

    return {"text": text, "hidden": hidden}


# ============================================================
# Тайминги
# ============================================================

class Timer:
    """Контекстный менеджер для замеров времени."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


def timeit(func, *args, **kwargs):
    """Вызывает func и возвращает (result, elapsed_seconds)."""
    t0 = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - t0
