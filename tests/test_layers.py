"""Tests for fist_lora.layers module."""

import pytest
import torch
import torch.nn as nn


class TestFiSTLoRALinear:
    def test_output_shape(self):
        from fist_lora.layers import FiSTLoRALinear

        d, k, r = 64, 32, 8
        original = nn.Linear(k, d)
        B = torch.randn(d, r)
        A = torch.randn(r, k)
        R = torch.zeros(r, r)

        layer = FiSTLoRALinear(original, B, A, R, alpha=32.0, rank=r)
        x = torch.randn(4, k)
        out = layer(x)

        assert out.shape == (4, d)

    def test_3d_input(self):
        from fist_lora.layers import FiSTLoRALinear

        d, k, r = 64, 32, 8
        original = nn.Linear(k, d)
        B = torch.randn(d, r)
        A = torch.randn(r, k)
        R = torch.zeros(r, r)

        layer = FiSTLoRALinear(original, B, A, R, alpha=32.0, rank=r)
        x = torch.randn(2, 10, k)  # batch=2, seq=10
        out = layer(x)

        assert out.shape == (2, 10, d)

    def test_zero_R_equals_base(self):
        from fist_lora.layers import FiSTLoRALinear

        d, k, r = 64, 32, 8
        original = nn.Linear(k, d)
        B = torch.randn(d, r)
        A = torch.randn(r, k)
        R = torch.zeros(r, r)

        layer = FiSTLoRALinear(original, B, A, R, alpha=32.0, rank=r)
        x = torch.randn(4, k)

        base_out = torch.nn.functional.linear(x, original.weight, original.bias)
        fist_out = layer(x)

        assert torch.allclose(base_out, fist_out, atol=1e-5), \
            "With zero R, FiSTLoRALinear should equal the base linear"

    def test_nonzero_R_differs(self):
        from fist_lora.layers import FiSTLoRALinear

        d, k, r = 64, 32, 8
        original = nn.Linear(k, d)
        B = torch.randn(d, r)
        A = torch.randn(r, k)
        R = torch.randn(r, r) * 0.01

        layer = FiSTLoRALinear(original, B, A, R, alpha=32.0, rank=r)
        x = torch.randn(4, k)

        base_out = torch.nn.functional.linear(x, original.weight, original.bias)
        fist_out = layer(x)

        diff = (base_out - fist_out).abs().max()
        assert diff > 1e-6, "With nonzero R, output should differ from base"

    def test_R_is_trainable(self):
        from fist_lora.layers import FiSTLoRALinear

        d, k, r = 64, 32, 8
        original = nn.Linear(k, d)
        B = torch.randn(d, r)
        A = torch.randn(r, k)
        R = torch.randn(r, r) * 0.01

        layer = FiSTLoRALinear(original, B, A, R, alpha=32.0, rank=r)

        assert layer.R.requires_grad

    def test_weight_is_frozen(self):
        from fist_lora.layers import FiSTLoRALinear

        original = nn.Linear(32, 64)
        layer = FiSTLoRALinear(
            original,
            torch.randn(64, 8),
            torch.randn(8, 32),
            torch.zeros(8, 8),
            alpha=32.0,
            rank=8,
        )

        assert not layer.weight.requires_grad

    def test_B_A_are_buffers(self):
        from fist_lora.layers import FiSTLoRALinear

        original = nn.Linear(32, 64)
        layer = FiSTLoRALinear(
            original,
            torch.randn(64, 8),
            torch.randn(8, 32),
            torch.zeros(8, 8),
            alpha=32.0,
            rank=8,
        )

        # B and A should be buffers, not parameters
        param_names = {name for name, _ in layer.named_parameters()}
        assert "B" not in param_names
        assert "A" not in param_names

        buffer_names = {name for name, _ in layer.named_buffers()}
        assert "B" in buffer_names
        assert "A" in buffer_names

    def test_scaling_factor(self):
        from fist_lora.layers import FiSTLoRALinear

        alpha, rank = 32.0, 8
        original = nn.Linear(32, 64)
        layer = FiSTLoRALinear(
            original,
            torch.randn(64, rank),
            torch.randn(rank, 32),
            torch.zeros(rank, rank),
            alpha=alpha,
            rank=rank,
        )

        assert layer.scaling == alpha / rank

    def test_bias_handling_with_bias(self):
        from fist_lora.layers import FiSTLoRALinear

        original = nn.Linear(32, 64, bias=True)
        layer = FiSTLoRALinear(
            original,
            torch.randn(64, 8),
            torch.randn(8, 32),
            torch.zeros(8, 8),
            alpha=32.0,
            rank=8,
        )

        assert layer.bias is not None
        assert not layer.bias.requires_grad

    def test_bias_handling_without_bias(self):
        from fist_lora.layers import FiSTLoRALinear

        original = nn.Linear(32, 64, bias=False)
        layer = FiSTLoRALinear(
            original,
            torch.randn(64, 8),
            torch.randn(8, 32),
            torch.zeros(8, 8),
            alpha=32.0,
            rank=8,
        )

        assert layer.bias is None

    def test_gradient_flows_through_R(self):
        from fist_lora.layers import FiSTLoRALinear

        original = nn.Linear(32, 64)
        R = torch.randn(8, 8) * 0.01
        layer = FiSTLoRALinear(
            original,
            torch.randn(64, 8),
            torch.randn(8, 32),
            R,
            alpha=32.0,
            rank=8,
        )

        x = torch.randn(4, 32)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.R.grad is not None
        assert layer.R.grad.norm() > 0
