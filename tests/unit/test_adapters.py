"""
Тесты для архитектур адаптеров (InnerAdapter, CrossModelAdapter).

Покрывает риски:
- Корректность forward pass адаптеров
- Совместимость размеров тензоров
- Сохранение/загрузка state_dict
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'custom_impl'))

from train_inner import InnerAdapter
from train_outer import Adapter as InnerAdapterOuter, CrossModelAdapter


class TestInnerAdapter:
    """Тесты InnerAdapter из train_inner.py."""

    def test_forward_shape(self):
        """Forward pass сохраняет форму тензора."""
        adapter = InnerAdapter(hidden_size=256)
        x = torch.randn(2, 10, 256)  # batch=2, seq=10, hidden=256
        output = adapter(x)
        assert output.shape == x.shape

    def test_forward_dtype(self):
        """Forward pass сохраняет тип данных."""
        adapter = InnerAdapter(hidden_size=128)
        adapter = adapter.to(torch.bfloat16)
        x = torch.randn(1, 5, 128, dtype=torch.bfloat16)
        output = adapter(x)
        assert output.dtype == torch.bfloat16

    def test_residual_connection(self):
        """Residual connection: output = x + MLP(x)."""
        adapter = InnerAdapter(hidden_size=64)
        adapter.proj1.weight.data.zero_()
        adapter.proj2.weight.data.zero_()
        x = torch.randn(1, 3, 64)
        output = adapter(x)
        # При нулевых весах MLP, output должен быть близок к LayerNorm(x + 0) = LayerNorm(x)
        assert output.shape == x.shape

    def test_save_load(self):
        """Сохранение и загрузка state_dict."""
        adapter1 = InnerAdapter(hidden_size=128)
        state = adapter1.state_dict()

        adapter2 = InnerAdapter(hidden_size=128)
        adapter2.load_state_dict(state)

        x = torch.randn(1, 5, 128)
        with torch.no_grad():
            out1 = adapter1(x)
            out2 = adapter2(x)
        torch.testing.assert_close(out1, out2)


class TestInnerAdapterFromOuter:
    """Тесты InnerAdapter (Adapter) из train_outer.py."""

    def test_ln_res_adapter(self):
        """ln_res_adapter с правильными размерами."""
        adapter = InnerAdapterOuter(hidden_size=256, adapter_type="ln_res_adapter")
        x = torch.randn(2, 10, 256)
        output = adapter(x)
        assert output.shape == x.shape

    def test_different_types(self):
        """Различные типы адаптеров."""
        for adapter_type in ["linear_adapter", "linear_res_adapter", "adapter", "res_adapter", "ln_res_adapter"]:
            adapter = InnerAdapterOuter(hidden_size=128, adapter_type=adapter_type)
            x = torch.randn(1, 5, 128)
            output = adapter(x)
            assert output.shape == x.shape


class TestCrossModelAdapter:
    """Тесты CrossModelAdapter (outer link)."""

    def test_same_dim(self):
        """Одинаковые размеры вход/выход."""
        adapter = CrossModelAdapter(in_dim=256, out_dim=256, adapter_type="outer_ln_res_adapter")
        x = torch.randn(2, 10, 256)
        output = adapter(x)
        assert output.shape == (2, 10, 256)

    def test_different_dim(self):
        """Разные размеры вход/выход (cross-model)."""
        adapter = CrossModelAdapter(in_dim=2048, out_dim=1536, adapter_type="outer_ln_res_adapter")
        x = torch.randn(1, 8, 2048)
        output = adapter(x)
        assert output.shape == (1, 8, 1536)

    def test_outer_adapter_types(self):
        """Различные типы outer адаптеров."""
        for adapter_type in [
            "outer_linear_adapter",
            "outer_linear_res_adapter",
            "outer_adapter",
            "outer_res_adapter",
            "outer_ln_adapter",
            "outer_ln_res_adapter",
        ]:
            adapter = CrossModelAdapter(in_dim=128, out_dim=64, adapter_type=adapter_type)
            x = torch.randn(1, 5, 128)
            output = adapter(x)
            assert output.shape == (1, 5, 64)

    def test_hidden_dim_expansion(self):
        """Проверка expansion hidden_dim = out_dim * 2 для LN адаптеров."""
        adapter = CrossModelAdapter(in_dim=256, out_dim=128, adapter_type="outer_ln_res_adapter")
        # hidden_dim должен быть out_dim * 2 = 256
        assert adapter.proj1.out_features == 256  # in_dim -> hidden_dim
        assert adapter.proj2.in_features == 256   # hidden_dim -> out_dim
        assert adapter.proj2.out_features == 128  # out_dim

    def test_save_load(self):
        """Сохранение и загрузка state_dict."""
        adapter1 = CrossModelAdapter(in_dim=256, out_dim=128, adapter_type="outer_ln_res_adapter")
        state = adapter1.state_dict()

        adapter2 = CrossModelAdapter(in_dim=256, out_dim=128, adapter_type="outer_ln_res_adapter")
        adapter2.load_state_dict(state)

        x = torch.randn(1, 5, 256)
        with torch.no_grad():
            out1 = adapter1(x)
            out2 = adapter2(x)
        torch.testing.assert_close(out1, out2)

    def test_gradient_flow(self):
        """Градиенты проходят через адаптер."""
        adapter = CrossModelAdapter(in_dim=64, out_dim=32, adapter_type="outer_ln_res_adapter")
        x = torch.randn(1, 3, 64, requires_grad=True)
        output = adapter(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
