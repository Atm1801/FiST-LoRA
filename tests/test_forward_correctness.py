"""
End-to-end forward correctness tests for FiST-LoRA.

Verifies that:
1. inject_fist_lora correctly replaces layers
2. Trainable param count matches expectations
3. Forward pass produces correct output
4. Only R matrices and head are trainable
"""

import pytest
import torch
import torch.nn as nn


def make_transformer_like_model(d=64, num_layers=2, num_classes=2):
    """Create a simplified transformer-like model for testing.
    Uses d for both in and out features so residual connections work."""

    class TransformerLayer(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.query = nn.Linear(d, d)
            self.key = nn.Linear(d, d)
            self.value = nn.Linear(d, d)
            self.dense = nn.Linear(d, d)

        def forward(self, x):
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            return self.dense(q + k + v)

    class SimpleTransformer(nn.Module):
        def __init__(self, d, num_layers, num_classes):
            super().__init__()
            self.layers = nn.ModuleList([
                TransformerLayer(d) for _ in range(num_layers)
            ])
            self.classifier = nn.Linear(d, num_classes)

        def forward(self, input_ids=None, labels=None, **kwargs):
            x = input_ids.float()
            for layer in self.layers:
                x = x + layer(x)
            logits = self.classifier(x.mean(dim=1) if x.dim() == 3 else x)

            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)

            class Output:
                pass
            o = Output()
            o.loss = loss
            o.logits = logits
            return o

    return SimpleTransformer(d, num_layers, num_classes)


class TestInjectFistLoRA:
    def test_replaces_target_modules(self):
        from fist_lora.layers import FiSTLoRALinear
        from fist_lora.model import inject_fist_lora, collect_plain_svd
        from fist_lora.init import zero_R

        model = make_transformer_like_model()
        rank = 4

        svd = collect_plain_svd(model, ["query", "key", "value"], rank)
        B_dict = {n: BA[0] for n, BA in svd.items()}
        A_dict = {n: BA[2] for n, BA in svd.items()}
        R_dict = {n: zero_R(rank) for n in svd}

        model = inject_fist_lora(model, ["query", "key", "value"], B_dict, A_dict, R_dict, 32.0, rank)

        # Check that target modules are replaced
        fist_count = 0
        for name, module in model.named_modules():
            if isinstance(module, FiSTLoRALinear):
                fist_count += 1

        # 2 layers * 3 modules = 6
        assert fist_count == 6, f"Expected 6 FiSTLoRALinear modules, got {fist_count}"

    def test_dense_not_replaced(self):
        from fist_lora.layers import FiSTLoRALinear
        from fist_lora.model import inject_fist_lora, collect_plain_svd
        from fist_lora.init import zero_R

        model = make_transformer_like_model()
        rank = 4

        svd = collect_plain_svd(model, ["query", "key", "value"], rank)
        B_dict = {n: BA[0] for n, BA in svd.items()}
        A_dict = {n: BA[2] for n, BA in svd.items()}
        R_dict = {n: zero_R(rank) for n in svd}

        model = inject_fist_lora(model, ["query", "key", "value"], B_dict, A_dict, R_dict, 32.0, rank)

        # Dense layers should NOT be replaced
        for name, module in model.named_modules():
            if "dense" in name and isinstance(module, nn.Linear):
                assert not isinstance(module, FiSTLoRALinear)

    def test_trainable_params_correct(self):
        from fist_lora.model import inject_fist_lora, count_trainable_params, collect_plain_svd
        from fist_lora.init import zero_R

        d = 64
        num_layers = 2
        num_classes = 2
        rank = 4

        model = make_transformer_like_model(d, num_layers, num_classes)

        svd = collect_plain_svd(model, ["query", "key", "value"], rank)
        B_dict = {n: BA[0] for n, BA in svd.items()}
        A_dict = {n: BA[2] for n, BA in svd.items()}
        R_dict = {n: zero_R(rank) for n in svd}

        model = inject_fist_lora(model, ["query", "key", "value"], B_dict, A_dict, R_dict, 32.0, rank)

        trainable = count_trainable_params(model)

        # Expected: 6 R matrices (rank^2 each) + classifier (d * num_classes + num_classes)
        n_R_modules = num_layers * 3  # query, key, value per layer
        expected_R_params = n_R_modules * (rank ** 2)
        expected_head_params = d * num_classes + num_classes
        expected_total = expected_R_params + expected_head_params

        assert trainable == expected_total, \
            f"Expected {expected_total} trainable params, got {trainable}"

    def test_only_R_and_head_trainable(self):
        from fist_lora.model import inject_fist_lora, collect_plain_svd
        from fist_lora.init import zero_R
        from fist_lora.layers import FiSTLoRALinear

        model = make_transformer_like_model()
        rank = 4

        svd = collect_plain_svd(model, ["query", "key", "value"], rank)
        B_dict = {n: BA[0] for n, BA in svd.items()}
        A_dict = {n: BA[2] for n, BA in svd.items()}
        R_dict = {n: zero_R(rank) for n in svd}

        model = inject_fist_lora(model, ["query", "key", "value"], B_dict, A_dict, R_dict, 32.0, rank)

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert "R" in name or "classifier" in name, \
                    f"Unexpected trainable parameter: {name}"


class TestForwardCorrectness:
    def test_forward_with_zero_R(self):
        from fist_lora.model import inject_fist_lora, collect_plain_svd
        from fist_lora.init import zero_R

        model = make_transformer_like_model()
        rank = 4

        # Get baseline output
        x = torch.randn(4, 64)
        with torch.no_grad():
            baseline_out = model(input_ids=x).logits.clone()

        svd = collect_plain_svd(model, ["query", "key", "value"], rank)
        B_dict = {n: BA[0] for n, BA in svd.items()}
        A_dict = {n: BA[2] for n, BA in svd.items()}
        R_dict = {n: zero_R(rank) for n in svd}

        model = inject_fist_lora(model, ["query", "key", "value"], B_dict, A_dict, R_dict, 32.0, rank)

        with torch.no_grad():
            adapted_out = model(input_ids=x).logits

        assert torch.allclose(baseline_out, adapted_out, atol=1e-4), \
            "With zero R, adapted model should produce same output as base"

    def test_forward_produces_loss(self):
        from fist_lora.model import inject_fist_lora, collect_plain_svd
        from fist_lora.init import zero_R

        model = make_transformer_like_model()
        rank = 4

        svd = collect_plain_svd(model, ["query", "key", "value"], rank)
        B_dict = {n: BA[0] for n, BA in svd.items()}
        A_dict = {n: BA[2] for n, BA in svd.items()}
        R_dict = {n: zero_R(rank) for n in svd}

        model = inject_fist_lora(model, ["query", "key", "value"], B_dict, A_dict, R_dict, 32.0, rank)

        x = torch.randn(4, 64)
        labels = torch.tensor([0, 1, 0, 1])
        output = model(input_ids=x, labels=labels)

        assert output.loss is not None
        assert output.loss.requires_grad  # Should be differentiable through R

    def test_backward_updates_R(self):
        from fist_lora.model import inject_fist_lora, collect_plain_svd
        from fist_lora.init import zero_R
        from fist_lora.layers import FiSTLoRALinear

        model = make_transformer_like_model()
        rank = 4

        svd = collect_plain_svd(model, ["query", "key", "value"], rank)
        B_dict = {n: BA[0] for n, BA in svd.items()}
        A_dict = {n: BA[2] for n, BA in svd.items()}
        R_dict = {n: torch.randn(rank, rank) * 0.01 for n in svd}

        model = inject_fist_lora(model, ["query", "key", "value"], B_dict, A_dict, R_dict, 32.0, rank)

        x = torch.randn(4, 64)
        labels = torch.tensor([0, 1, 0, 1])

        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )

        # Save initial R values
        initial_Rs = {}
        for name, module in model.named_modules():
            if isinstance(module, FiSTLoRALinear):
                initial_Rs[name] = module.R.data.clone()

        # One training step
        output = model(input_ids=x, labels=labels)
        output.loss.backward()
        optimizer.step()

        # Check R values changed
        for name, module in model.named_modules():
            if isinstance(module, FiSTLoRALinear):
                diff = (module.R.data - initial_Rs[name]).abs().max()
                assert diff > 0, f"R in {name} should have been updated"

    def test_frozen_weights_unchanged(self):
        from fist_lora.model import inject_fist_lora, collect_plain_svd
        from fist_lora.init import zero_R
        from fist_lora.layers import FiSTLoRALinear

        model = make_transformer_like_model()
        rank = 4

        svd = collect_plain_svd(model, ["query", "key", "value"], rank)
        B_dict = {n: BA[0] for n, BA in svd.items()}
        A_dict = {n: BA[2] for n, BA in svd.items()}
        R_dict = {n: torch.randn(rank, rank) * 0.01 for n in svd}

        model = inject_fist_lora(model, ["query", "key", "value"], B_dict, A_dict, R_dict, 32.0, rank)

        # Save initial frozen weights
        initial_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, FiSTLoRALinear):
                initial_weights[name] = {
                    "weight": module.weight.data.clone(),
                    "B": module.B.clone(),
                    "A": module.A.clone(),
                }

        # Training step
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        x = torch.randn(4, 64)
        labels = torch.tensor([0, 1, 0, 1])
        output = model(input_ids=x, labels=labels)
        output.loss.backward()
        optimizer.step()

        # Verify frozen weights unchanged
        for name, module in model.named_modules():
            if isinstance(module, FiSTLoRALinear):
                assert torch.equal(module.weight.data, initial_weights[name]["weight"])
                assert torch.equal(module.B, initial_weights[name]["B"])
                assert torch.equal(module.A, initial_weights[name]["A"])
