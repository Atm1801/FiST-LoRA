"""Tests for fist_lora.fisher module."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_simple_model(d=64, k=32, num_classes=2):
    """Create a simple model with named linear layers for testing."""
    model = nn.Sequential()
    model.add_module("query", nn.Linear(k, d))
    model.add_module("key", nn.Linear(k, d))
    model.add_module("value", nn.Linear(k, d))
    model.add_module("classifier", nn.Linear(d, num_classes))

    class WrappedModel(nn.Module):
        def __init__(self, seq):
            super().__init__()
            self.query = seq.query
            self.key = seq.key
            self.value = seq.value
            self.classifier = seq.classifier

        def forward(self, input_ids=None, labels=None, **kwargs):
            x = input_ids.float()
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            out = q + k + v
            logits = self.classifier(out.mean(dim=1) if out.dim() == 3 else out)

            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)

            class Output:
                pass
            output = Output()
            output.loss = loss
            output.logits = logits
            return output

    return WrappedModel(model)


def make_dataloader(batch_size=8, num_batches=4, k=32, num_classes=2):
    """Create a simple dataloader with random data."""
    total = batch_size * num_batches
    input_ids = torch.randn(total, k)
    labels = torch.randint(0, num_classes, (total,))
    dataset = TensorDataset(input_ids, labels)

    def collate_fn(batch):
        inputs, labs = zip(*batch)
        return {"input_ids": torch.stack(inputs), "labels": torch.stack(labs)}

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


class TestComputeDiagonalFisher:
    def test_returns_correct_keys(self):
        from fist_lora.fisher import compute_diagonal_fisher

        model = make_simple_model()
        loader = make_dataloader()

        fisher = compute_diagonal_fisher(model, loader, ["query", "key", "value"], num_samples=16)

        assert "query" in fisher
        assert "key" in fisher
        assert "value" in fisher
        assert "classifier" not in fisher

    def test_output_shapes(self):
        from fist_lora.fisher import compute_diagonal_fisher

        d, k = 64, 32
        model = make_simple_model(d=d, k=k)
        loader = make_dataloader(k=k)

        fisher = compute_diagonal_fisher(model, loader, ["query", "key"], num_samples=16)

        for name, f_diag in fisher.items():
            assert f_diag.shape == (d, k), f"Expected ({d}, {k}), got {f_diag.shape}"

    def test_non_negative(self):
        from fist_lora.fisher import compute_diagonal_fisher

        model = make_simple_model()
        loader = make_dataloader()

        fisher = compute_diagonal_fisher(model, loader, ["query"], num_samples=16)

        for f_diag in fisher.values():
            assert (f_diag >= 0).all(), "Fisher diagonal should be non-negative"

    def test_nonzero(self):
        from fist_lora.fisher import compute_diagonal_fisher

        model = make_simple_model()
        loader = make_dataloader()

        fisher = compute_diagonal_fisher(model, loader, ["query"], num_samples=16)

        for f_diag in fisher.values():
            assert f_diag.sum() > 0, "Fisher should have at least some non-zero entries"

    def test_requires_grad_restored(self):
        from fist_lora.fisher import compute_diagonal_fisher

        model = make_simple_model()
        loader = make_dataloader()

        # Ensure weights start without grad
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.requires_grad_(False)

        compute_diagonal_fisher(model, loader, ["query", "key"], num_samples=16)

        # Check grads are restored
        assert not model.query.weight.requires_grad
        assert not model.key.weight.requires_grad

    def test_no_matching_modules_raises(self):
        from fist_lora.fisher import compute_diagonal_fisher

        model = make_simple_model()
        loader = make_dataloader()

        with pytest.raises(ValueError, match="No Linear modules"):
            compute_diagonal_fisher(model, loader, ["nonexistent_module"], num_samples=16)

    def test_num_samples_limit(self):
        from fist_lora.fisher import compute_diagonal_fisher

        model = make_simple_model()
        loader = make_dataloader(batch_size=8, num_batches=10)

        # With num_samples=8, should process at most 1 batch
        fisher = compute_diagonal_fisher(model, loader, ["query"], num_samples=8)

        # Should still produce valid output
        assert len(fisher) == 1
        assert fisher["query"].sum() > 0
