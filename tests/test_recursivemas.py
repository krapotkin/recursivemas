#!/usr/bin/env python3
"""Тесты для проекта RecursiveMAS.

Запуск:
  pytest tests/test_recursivemas.py -v
  pytest tests/test_recursivemas.py -v -k dataset   # только тесты датасетов
  pytest tests/test_recursivemas.py -v -k parse     # только парсинг
"""

import os
import sys
import json
import re
import tempfile
from pathlib import Path

import pytest
import torch
import importlib
import importlib.util
import importlib.machinery

# Добавляем custom_impl в путь
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "custom_impl"))


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


# ============================================================
# Тесты: dataset_utils — парсинг ответов
# ============================================================

class TestParseAnswer:
    """Парсинг ответов из текста модели — самая частая ошибка."""

    def setup_method(self):
        from dataset_utils import parse_answer, normalize_answer, _parse_gt_answer
        self.parse = parse_answer
        self.normalize = normalize_answer
        self.parse_gt = _parse_gt_answer

    # --- parse_answer (из текста модели) ---

    def test_gsm8k_format(self):
        """Формат #### 42"""
        assert self.parse("Result is #### 42") == "42"

    def test_gsm8k_format_decimal(self):
        """Формат #### 3.5"""
        assert self.parse("Answer #### 3.5") == "3.5"

    def test_boxed_format(self):
        """Формат \\boxed{42}"""
        assert self.parse(r"Final answer \boxed{42}") == "42"

    def test_boxed_format_decimal(self):
        """Формат \\boxed{3.14}"""
        assert self.parse(r"Result \boxed{3.14}") == "3.14"

    def test_boxed_format_negative(self):
        """Формат \\boxed{-5}"""
        assert self.parse(r"Answer \boxed{-5}") == "-5"

    def test_last_number_fallback(self):
        """Текст без чисел — возвращаем None."""
        assert self.parse("The answer is forty two") is None  # нет цифр
        assert self.parse("I think the result is 42") == "42"

    def test_last_number_with_text(self):
        """Последнее число среди текста."""
        assert self.parse("First 10, then 20, finally 30") == "30"

    def test_empty_text(self):
        """Пустой текст -> None."""
        assert self.parse("") is None
        assert self.parse(None) is None

    def test_comma_in_number(self):
        """Числа с запятыми."""
        result = self.parse("Answer #### 1,200")
        assert result == "1200"  # запятая удаляется

    def test_dollar_sign(self):
        """Знак доллара."""
        result = self.parse("Result #### $42")
        assert result == "42"

    # --- normalize_answer ---

    def test_normalize_plain(self):
        assert self.normalize("42") == "42"

    def test_normalize_decimal(self):
        assert self.normalize("3.14") == "3.14"

    def test_normalize_comma(self):
        assert self.normalize("1,200") == "1200"

    def test_normalize_dollar(self):
        assert self.normalize("$42") == "42"

    def test_normalize_trailing_dot(self):
        assert self.normalize("42.") == "42"

    # --- _parse_gt_answer (parsing ground truth) ---

    def test_parse_gt_gsm8k(self):
        """GSM8K формат: текст #### число."""
        assert self.parse_gt("She makes 9 * 2 = $18\n#### 18", "gsm8k") == "18"

    def test_parse_gt_gsm8k_complex(self):
        """GSM8K с длинным решением."""
        raw = "First find 200 * 0.4 = 80 GB\nThen 200 - 80 = 120 GB\n#### 120"
        assert self.parse_gt(raw, "gsm8k") == "120"

    def test_parse_gt_gsm8k_fallback(self):
        """GSM8K без #### — последнее число."""
        assert self.parse_gt("The answer is 42", "gsm8k") == "42"

    def test_parse_gt_math500_plain(self):
        """Math500: просто число."""
        assert self.parse_gt("42", "math500") == "42"

    def test_parse_gt_math500_fraction(self):
        """Math500 с дробью."""
        assert self.parse_gt("3/4", "math500") == "3/4"  # не число — возвращаем как есть
        # Проверим что оно не упало
        assert isinstance(self.parse_gt("3/4", "math500"), str)

    def test_parse_gt_empty(self):
        """Пустой ответ."""
        assert self.parse_gt("", "gsm8k") == ""
        assert self.parse_gt(None, "gsm8k") == ""


# ============================================================
# Тесты: dataset_utils — compute_accuracy
# ============================================================

class TestComputeAccuracy:
    def setup_method(self):
        from dataset_utils import compute_accuracy
        self.compute = compute_accuracy

    def test_all_correct(self):
        result = self.compute(
            ["Answer #### 42", "Result #### 18"],
            ["42", "18"],
        )
        assert result["accuracy"] == 1.0
        assert result["correct"] == 2
        assert result["total"] == 2

    def test_all_wrong(self):
        result = self.compute(
            ["Answer #### 42", "Result #### 18"],
            ["100", "200"],
        )
        assert result["accuracy"] == 0.0
        assert result["correct"] == 0

    def test_mixed(self):
        result = self.compute(
            ["Answer #### 42", "Result #### 18"],
            ["42", "99"],
        )
        assert result["accuracy"] == 0.5
        assert result["correct"] == 1

    def test_boxed_comparison(self):
        """\\boxed{42} vs 42 — должно совпадать."""
        result = self.compute(
            [r"Answer \boxed{42}"],
            ["42"],
        )
        assert result["correct"] == 1

    def test_normalization(self):
        """Запятые не должны влиять."""
        result = self.compute(
            ["Answer #### $1,200"],
            ["1200"],
        )
        assert result["correct"] == 1


# ============================================================
# Тесты: dataset_utils — load_dataset_safe
# ============================================================

class TestLoadDataset:
    def setup_method(self):
        from dataset_utils import load_dataset_safe
        self.load = load_dataset_safe

    def test_load_gsm8k_small(self):
        """Загрузка 5 примеров GSM8K."""
        data = self.load("gsm8k", num_examples=5, split="train")
        assert len(data["questions"]) == 5
        assert len(data["answers"]) == 5
        assert len(data["gt_numbers"]) == 5
        assert data["dataset_name"] == "gsm8k"

    def test_gsm8k_has_correct_format(self):
        """GSM8K ответы содержат ####."""
        data = self.load("gsm8k", num_examples=5, split="train")
        for ans in data["answers"]:
            assert "####" in ans, f"GSM8K answer lacks ####: {ans[:50]}"

    def test_gsm8k_gt_parsed(self):
        """GSM8K gt_numbers — числа."""
        data = self.load("gsm8k", num_examples=5, split="train")
        for gt in data["gt_numbers"]:
            assert gt != "", f"Empty gt_number"
            # Может быть числом или выражением
            assert isinstance(gt, str)

    def test_load_math500_small(self):
        """Загрузка 5 примеров Math500."""
        data = self.load("math500", num_examples=5)
        assert len(data["questions"]) == 5
        assert len(data["answers"]) == 5
        assert data["dataset_name"] == "math500"

    def test_load_invalid_dataset(self):
        """Неизвестный датасет -> ValueError."""
        from dataset_utils import load_dataset_safe
        with pytest.raises(ValueError, match="Неизвестный датасет"):
            load_dataset_safe("invalid_dataset")

    def test_math500_correct_split(self):
        """Math500 использует split test."""
        data = self.load("math500", num_examples=3)
        assert len(data["questions"]) == 3
        assert all(isinstance(q, str) for q in data["questions"])
        assert all(isinstance(a, str) for a in data["answers"])


# ============================================================
# Тесты: CrossModelAdapter
# ============================================================

class TestCrossModelAdapter:
    def setup_method(self):
        """Импортируем CrossModelAdapter из train_outer.py."""
        import importlib
        spec = importlib.util.find_spec("train_outer")
        if spec is None:
            # Загружаем как модуль из файла
            import types
            loader = importlib.machinery.SourceFileLoader(
                "train_outer",
                str(Path(__file__).resolve().parent.parent / "custom_impl" / "train_outer.py")
            )
            spec = importlib.util.spec_from_loader("train_outer", loader)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_create_adapter(self, device):
        """Создание CrossModelAdapter с разными размерностями."""
        adapter = self.mod.CrossModelAdapter(1536, 2048)
        assert adapter.in_dim == 1536
        assert adapter.out_dim == 2048
        # Проверяем что в нём есть параметры
        params = sum(p.numel() for p in adapter.parameters())
        assert params > 0, "Adapter has no parameters"

    def test_forward_pass(self, device):
        """Forward pass с правильными размерностями."""
        adapter = self.mod.CrossModelAdapter(1536, 2048).to(device)
        x = torch.randn(1, 32, 1536, device=device)
        out = adapter(x)
        assert out.shape == (1, 32, 2048), f"Expected (1,32,2048), got {out.shape}"

    def test_forward_planner_to_refiner(self, device):
        """Planner (1536) -> Refiner (2048), затем Refiner (2048) -> Solver (1536)."""
        outer_12 = self.mod.CrossModelAdapter(1536, 2048).to(device)
        outer_23 = self.mod.CrossModelAdapter(2048, 1536).to(device)

        planner_latent = torch.randn(1, 32, 1536, device=device)
        refiner_input = outer_12(planner_latent)
        solver_input = outer_23(refiner_input)

        assert refiner_input.shape == (1, 32, 2048)
        assert solver_input.shape == (1, 32, 1536)

    def test_init_aligned(self, device):
        """init_aligned с bridge матрицей."""
        adapter = self.mod.CrossModelAdapter(1536, 2048).to(device)
        bridge = torch.randn(2048, 1536)

        # Сохраняем веса до
        w_before = adapter.residual_proj.weight.clone()

        adapter.init_aligned(bridge)

        # Проверяем что веса изменились
        assert not torch.allclose(w_before, adapter.residual_proj.weight)

        # Проверяем что bridge скопирован правильно
        assert torch.allclose(
            adapter.residual_proj.weight.cpu(),
            bridge.to(adapter.residual_proj.weight.dtype)
        )

    def test_init_aligned_no_residual(self, device):
        """init_aligned без residual — не падает."""
        adapter = self.mod.CrossModelAdapter(1536, 2048,
                                             adapter_type="outer_ln_adapter").to(device)
        bridge = torch.randn(2048, 1536)
        # Просто не должно упасть
        adapter.init_aligned(bridge)

    def test_ln_res_adapter_type(self, device):
        """По умолчанию outer_ln_res_adapter."""
        adapter = self.mod.CrossModelAdapter(1536, 2048).to(device)
        assert adapter.use_ln is True
        assert adapter.use_residual is True
        assert adapter.residual_proj is not None
        assert adapter.ln_source is not None
        assert adapter.ln_target is not None


# ============================================================
# Тесты: Adapter
# ============================================================

class TestAdapter:
    def setup_method(self):
        import types
        from importlib.machinery import SourceFileLoader
        loader = SourceFileLoader(
            "train_outer",
            str(Path(__file__).resolve().parent.parent / "custom_impl" / "train_outer.py")
        )
        spec = importlib.util.spec_from_loader("train_outer", loader)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_create_adapter(self):
        """Создание Inner Adapter."""
        adapter = self.mod.Adapter(1536)
        params = sum(p.numel() for p in adapter.parameters())
        assert params > 0

    def test_forward_pass(self, device):
        """Forward pass через inner adapter."""
        adapter = self.mod.Adapter(1536).to(device)
        x = torch.randn(1, 32, 1536, device=device)
        out = adapter(x)
        assert out.shape == x.shape


# ============================================================
# Тесты: compute_procrustes_bridge
# ============================================================

class TestProcrustesBridge:
    def setup_method(self):
        import types
        from importlib.machinery import SourceFileLoader
        loader = SourceFileLoader(
            "train_outer",
            str(Path(__file__).resolve().parent.parent / "custom_impl" / "train_outer.py")
        )
        spec = importlib.util.spec_from_loader("train_outer", loader)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_exact_reconstruction(self, device):
        """Если H_src и H_tgt связаны линейно, bridge должен восстановить W почти точно."""
        n, d_in, d_out = 100, 32, 16
        H_src = torch.randn(n, d_in)
        W_true = torch.randn(d_out, d_in)
        H_tgt = H_src @ W_true.T + 0.01 * torch.randn(n, d_out)

        W_pred = self.mod.compute_procrustes_bridge(H_src, H_tgt, lam=0.01)

        assert W_pred.shape == (d_out, d_in)
        # MSE должен быть маленьким
        mse = (W_pred - W_true).pow(2).mean().item()
        assert mse < 0.1, f"MSE too large: {mse}"

    def test_shape_preservation(self):
        """Bridge сохраняет размерности."""
        H_src = torch.randn(50, 128)
        H_tgt = torch.randn(50, 256)
        W = self.mod.compute_procrustes_bridge(H_src, H_tgt)
        assert W.shape == (256, 128)

    def test_ridge_regression(self):
        """С регуляризацией bridge не вырождается, даже при коллинеарности."""
        H_src = torch.randn(30, 10)
        H_tgt = torch.randn(30, 5)
        W = self.mod.compute_procrustes_bridge(H_src, H_tgt, lam=1.0)
        assert W.shape == (5, 10)
        assert not torch.isnan(W).any()
        assert not torch.isinf(W).any()


# ============================================================
# Тесты: evaluate_recursivemas — load_outer форматы
# ============================================================

class TestOuterLoader:
    def test_load_outer_pt_files(self, tmp_path):
        """Загрузка outer_12.pt и outer_23.pt из директории."""
        # Создаём фейковые .pt файлы
        from evaluate_recursivemas import CrossModelAdapter

        adapter_12 = CrossModelAdapter(1536, 2048)
        adapter_23 = CrossModelAdapter(2048, 1536)

        torch.save(adapter_12.state_dict(), tmp_path / "outer_12.pt")
        torch.save(adapter_23.state_dict(), tmp_path / "outer_23.pt")

        # Проверяем что файлы созданы
        assert (tmp_path / "outer_12.pt").exists()
        assert (tmp_path / "outer_23.pt").exists()

        # Загружаем и проверяем
        state_12 = torch.load(tmp_path / "outer_12.pt", weights_only=True)
        loaded = CrossModelAdapter(1536, 2048)
        loaded.load_state_dict(state_12)
        assert loaded.in_dim == 1536
        assert loaded.out_dim == 2048


# ============================================================
# Тесты: train_outer — argparse флаги
# ============================================================

class TestTrainOuterArgs:
    def test_parse_args_defaults(self):
        """Проверяем что парсер создаётся и имеет флаги aligned init."""
        from train_outer import parse_args
        import sys

        # Сохраняем sys.argv и подменяем
        old_argv = sys.argv.copy()
        sys.argv = [
            "train_outer.py",
            "--agent1_model", "Qwen/Qwen3-1.7B",
            "--agent2_model", "Qwen/Qwen3-1.7B",
            "--agent3_model", "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "--agent1_inner", "/tmp/inner1",
            "--agent2_inner", "/tmp/inner2",
            "--agent3_inner", "/tmp/inner3",
            "--save_dir", "/tmp/save",
        ]
        try:
            args = parse_args()
            assert hasattr(args, "align_outer"), "Missing --align_outer flag"
            assert args.align_outer == 1, "Default align_outer should be 1"
            assert hasattr(args, "align_samples"), "Missing --align_samples flag"
            assert args.align_samples == 300, "Default align_samples should be 300"
        finally:
            sys.argv[:] = old_argv


# ============================================================
# Тесты: сквозные — gsm8k_utils (совместимость)
# ============================================================

class TestGsm8kCompat:
    """Проверяем что gsm8k_utils всё ещё работает (обратная совместимость)."""

    def test_parse_answer_from_gsm8k_utils(self):
        from gsm8k_utils import parse_answer
        assert parse_answer("#### 42") == "42"
        assert parse_answer(r"\boxed{3.14}") == "3.14"


# ============================================================
# Запуск: python tests/test_recursivemas.py
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])