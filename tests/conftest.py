"""
conftest.py — общие фикстуры для тестов проекта RecursiveMAS.
"""
import pytest
import sys
import os

# Добавляем custom_impl в путь для импортов
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'custom_impl'))


@pytest.fixture
def sample_questions():
    """Примеры вопросов GSM8K для тестирования."""
    return [
        "John has 5 apples. He buys 3 more. How many apples does he have?",
        "A train travels at 60 mph for 2 hours. How far does it travel?",
        "If a pizza costs $12 and you split it 4 ways, how much does each person pay?",
    ]


@pytest.fixture
def sample_answers():
    """Примеры ответов GSM8K."""
    return [
        "He has 5 + 3 = 8 apples. #### 8",
        "The train travels 60 * 2 = 120 miles. #### 120",
        "Each person pays 12 / 4 = $3. #### 3",
    ]
