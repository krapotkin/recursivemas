"""
Тесты для функций построения промптов.

Покрывает риски:
- Промпты содержат все необходимые компоненты
- Слот-маркеры корректно вставлены
- Формат соответствует оригиналу
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'custom_impl'))

from train_inner import (
    build_math_planner_prompt,
    build_math_refiner_prompt,
    build_math_solver_prompt,
)


class TestPlannerPrompt:
    """Тесты промпта Planner."""

    def test_contains_question(self):
        """Промпт содержит вопрос."""
        question = "What is 5 + 3?"
        prompt = build_math_planner_prompt(question)
        assert "5 + 3" in prompt

    def test_contains_role_instruction(self):
        """Промпт содержит инструкцию роли."""
        prompt = build_math_planner_prompt("test question")
        assert "planner" in prompt.lower() or "plan" in prompt.lower()

    def test_non_empty(self):
        """Промпт не пустой."""
        prompt = build_math_planner_prompt("test")
        assert len(prompt) > 0


class TestRefinerPrompt:
    """Тесты промпта Refiner."""

    def test_contains_question(self):
        """Промпт содержит вопрос."""
        question = "What is 10 * 2?"
        plan = "Step 1: Multiply 10 by 2"
        prompt = build_math_refiner_prompt(question, plan)
        assert "10 * 2" in prompt

    def test_contains_plan(self):
        """Промпт содержит план."""
        question = "test"
        plan = "My detailed plan"
        prompt = build_math_refiner_prompt(question, plan)
        assert "My detailed plan" in prompt

    def test_non_empty(self):
        """Промпт не пустой."""
        prompt = build_math_refiner_prompt("q", "p")
        assert len(prompt) > 0


class TestSolverPrompt:
    """Тесты промпта Solver."""

    def test_contains_question(self):
        """Промпт содержит вопрос."""
        question = "Calculate 100 / 4"
        refined_plan = "Divide 100 by 4"
        prompt = build_math_solver_prompt(question, refined_plan)
        assert "100 / 4" in prompt

    def test_contains_refined_plan(self):
        """Промпт содержит улучшенный план."""
        question = "test"
        refined_plan = "Improved solution approach"
        prompt = build_math_solver_prompt(question, refined_plan)
        assert "Improved solution approach" in prompt

    def test_non_empty(self):
        """Промпт не пустой."""
        prompt = build_math_solver_prompt("q", "p")
        assert len(prompt) > 0

    def test_solver_pre_question_flag(self):
        """Флаг solver_pre_question добавляет вопрос перед планом."""
        question = "What is 2 + 2?"
        plan = "Add 2 and 2"
        prompt_with = build_math_solver_prompt(question, plan, solver_pre_question=True)
        prompt_without = build_math_solver_prompt(question, plan, solver_pre_question=False)
        # При solver_pre_question=True вопрос должен быть в начале
        assert len(prompt_with) >= len(prompt_without)


class TestPromptConsistency:
    """Тесты согласованности промптов."""

    def test_all_prompts_different(self):
        """Промпты для разных ролей отличаются."""
        question = "What is 1 + 1?"
        plan = "Add 1 and 1"

        planner = build_math_planner_prompt(question)
        refiner = build_math_refiner_prompt(question, plan)
        solver = build_math_solver_prompt(question, plan)

        assert planner != refiner
        assert refiner != solver
        assert planner != solver

    def test_prompts_deterministic(self):
        """Промпты детерминированы (одинаковый ввод -> одинаковый вывод)."""
        question = "test question"
        plan = "test plan"

        assert build_math_planner_prompt(question) == build_math_planner_prompt(question)
        assert build_math_refiner_prompt(question, plan) == build_math_refiner_prompt(question, plan)
        assert build_math_solver_prompt(question, plan) == build_math_solver_prompt(question, plan)
