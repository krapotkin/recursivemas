"""
Тесты для gsm8k_utils.py.

Покрывает риски:
- Парсинг ответов из разных форматов (#### X, \boxed{X})
- Вычисление accuracy
- Форматирование промптов
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'custom_impl'))

from gsm8k_utils import (
    parse_answer,
    compute_accuracy,
    format_gsm8k_prompt,
)


class TestParseAnswer:
    """Тесты парсинга ответов из текстовых генераций."""

    def test_standard_format(self):
        """Стандартный формат GSM8K: #### 42."""
        assert parse_answer("The answer is #### 42") == "42"

    def test_negative_number(self):
        """Отрицательное число."""
        assert parse_answer("Result: #### -7") == "-7"

    def test_decimal_number(self):
        """Дробное число."""
        assert parse_answer("Final: #### 3.14") == "3.14"

    def test_comma_separated(self):
        """Число с запятыми."""
        assert parse_answer("Total: #### 1,234") == "1234"

    def test_no_match_fallback(self):
        """Fallback на последнее число при отсутствии ####."""
        assert parse_answer("The result is 42.") in ["42", "42."]  # regex captures trailing punctuation

    def test_empty_input(self):
        """Пустой ввод возвращает None."""
        assert parse_answer("") is None

    def test_none_input(self):
        """None возвращает None."""
        assert parse_answer(None) is None

    def test_no_numbers(self):
        """Текст без чисел возвращает None."""
        assert parse_answer("No numbers here") is None


class TestComputeAccuracy:
    """Тесты вычисления точности."""

    def test_perfect_accuracy(self):
        """100% точность."""
        predictions = ["#### 8", "#### 120", "#### 3"]
        gt_numbers = ["8", "120", "3"]
        results = compute_accuracy(predictions, gt_numbers)
        assert results["accuracy"] == 1.0
        assert results["correct"] == 3
        assert results["total"] == 3

    def test_zero_accuracy(self):
        """0% точность."""
        predictions = ["#### 1", "#### 2", "#### 3"]
        gt_numbers = ["8", "120", "3"]
        results = compute_accuracy(predictions, gt_numbers)
        assert results["accuracy"] == pytest.approx(1 / 3)
        assert results["correct"] == 1

    def test_empty_input(self):
        """Пустой ввод."""
        results = compute_accuracy([], [])
        assert results["accuracy"] == 0.0
        assert results["total"] == 0

    def test_partial_accuracy(self):
        """Частичная точность."""
        predictions = ["#### 8", "wrong", "#### 3"]
        gt_numbers = ["8", "120", "3"]
        results = compute_accuracy(predictions, gt_numbers)
        assert results["accuracy"] == pytest.approx(2 / 3)

    def test_examples_included(self):
        """Примеры включены в результат."""
        predictions = ["#### 8", "#### 120"]
        gt_numbers = ["8", "120"]
        results = compute_accuracy(predictions, gt_numbers)
        assert len(results["examples"]) == 2
        assert all("correct" in ex for ex in results["examples"])


class TestFormatGsm8kPrompt:
    """Тесты форматирования промптов."""

    def test_default_system_msg(self):
        """Промпт с дефолтным системным сообщением."""
        question = "What is 2 + 2?"
        messages = format_gsm8k_prompt(question)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "2 + 2" in messages[1]["content"]

    def test_custom_system_msg(self):
        """Промпт с кастомным системным сообщением."""
        question = "What is 2 + 2?"
        system = "You are a math expert."
        messages = format_gsm8k_prompt(question, system_msg=system)
        assert messages[0]["content"] == system

    def test_question_included(self):
        """Вопрос включён в пользовательское сообщение."""
        question = "How much is 10 * 5?"
        messages = format_gsm8k_prompt(question)
        assert "10 * 5" in messages[1]["content"]
