"""Tests for fist_lora.init module."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_simple_model_and_loader(d=64, k=32, num_classes=2, batch_size=8, num_batches=4):
    """Create model and dataloader for init tests."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(k, d)
            self.key = nn.Linear(k, d)
            self.classifier = nn.Linear(d, num_classes)

        def forward(self, input_ids=None, labels=None, **kwargs):
            x = input_ids.float()
            out = self.query(x) + self.key(x)
            logits = self.classifier(out)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)

            class Output:
                pass
            o = Output()
            o.loss = loss
            o.logits = logits
            return o

    model = SimpleModel()

    total = batch_size * num_batches
    input_ids = torch.randn(total, k)
    labels = torch.randint(0, num_classes, (total,))
    dataset = TensorDataset(input_ids, labels)

    def collate_fn(batch):
        inputs, labs = zip(*batch)
        return {"input_ids": torch.stack(inputs), "labels": torch.stack(labs)}

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return model, loader


class TestPlainSVD:
    def test_output_shapes(self):
        from fist_lora.init import plain_svd

        d, k, r = 64, 32, 8
        W = torch.randn(d, k)
        B, S, A = plain_svd(W, r)

        assert B.shape == (d, r)
        assert S.shape == (r,)
        assert A.shape == (r, k)

    def test_reconstruction_quality(self):
        from fist_lora.init import plain_svd

        d, k, r = 64, 32, 8
        W = torch.randn(d, k)
        B, S, A = plain_svd(W, r)

        W_approx = B @ torch.diag(S) @ A
        reconstruction_error = (W - W_approx).norm() / W.norm()

        # With rank 8 out of 32, should capture significant portion
        assert reconstruction_error < 1.0

    def test_orthonormality(self):
        from fist_lora.init import plain_svd

        W = torch.randn(64, 32)
        B, S, A = plain_svd(W, 8)

        # B columns should be approximately orthonormal
        identity_approx = B.T @ B
        assert torch.allclose(identity_approx, torch.eye(8), atol=1e-5)

        # A rows should be approximately orthonormal
        identity_approx = A @ A.T
        assert torch.allclose(identity_approx, torch.eye(8), atol=1e-5)


class TestFisherWeightedSVD:
    def test_output_shapes(self):
        from fist_lora.init import fisher_weighted_svd

        d, k, r = 64, 32, 8
        W = torch.randn(d, k)
        F = torch.rand(d, k).abs() + 1e-6
        B, S, A = fisher_weighted_svd(W, F, r)

        assert B.shape == (d, r)
        assert S.shape == (r,)
        assert A.shape == (r, k)

    def test_clipping_works(self):
        from fist_lora.init import fisher_weighted_svd

        W = torch.randn(64, 32)
        F = torch.ones(64, 32)
        F[0, 0] = 1e6  # extreme outlier

        # Should not crash and should produce finite output
        B, S, A = fisher_weighted_svd(W, F, 8)
        assert torch.isfinite(B).all()
        assert torch.isfinite(S).all()
        assert torch.isfinite(A).all()

    def test_differs_from_plain(self):
        from fist_lora.init import fisher_weighted_svd, plain_svd

        W = torch.randn(64, 32)
        F = torch.rand(64, 32).abs() * 10 + 0.1  # non-uniform Fisher

        B_plain, _, _ = plain_svd(W, 8)
        B_fisher, _, _ = fisher_weighted_svd(W, F, 8)

        # The subspaces should generally differ
        # (Not guaranteed but very likely with non-uniform Fisher)
        diff = (B_plain - B_fisher).norm()
        assert diff > 1e-4, "Fisher-SVD should generally differ from plain SVD"

    def test_uniform_fisher_approx_plain(self):
        from fist_lora.init import fisher_weighted_svd, plain_svd

        W = torch.randn(64, 32)
        F = torch.ones(64, 32)  # uniform Fisher

        B_plain, S_plain, A_plain = plain_svd(W, 8)
        B_fisher, S_fisher, A_fisher = fisher_weighted_svd(W, F, 8)

        # With uniform Fisher (after clipping/normalization, still uniform),
        # the result should be close to plain SVD
        # Check if singular values are similar (subspaces may have sign flips)
        assert torch.allclose(S_plain, S_fisher, atol=0.1)


class TestGradientProjectedR:
    def test_output_shapes(self):
        from fist_lora.init import gradient_projected_R, plain_svd

        model, loader = make_simple_model_and_loader()
        rank = 4

        W_q = model.query.weight.data.float()
        B, S, A = plain_svd(W_q, rank)

        frozen_BA = {"query": (B, A)}
        R_dict = gradient_projected_R(
            model, loader, frozen_BA, ["query"],
            num_samples=16, rank=rank, init_scale=0.01,
        )

        assert "query" in R_dict
        assert R_dict["query"].shape == (rank, rank)

    def test_init_scale(self):
        from fist_lora.init import gradient_projected_R, plain_svd

        model, loader = make_simple_model_and_loader()
        rank = 4
        init_scale = 0.01

        W_q = model.query.weight.data.float()
        B, S, A = plain_svd(W_q, rank)

        frozen_BA = {"query": (B, A)}
        R_dict = gradient_projected_R(
            model, loader, frozen_BA, ["query"],
            num_samples=16, rank=rank, init_scale=init_scale,
        )

        R = R_dict["query"]
        norm = R.norm().item()
        assert abs(norm - init_scale) < 1e-4, f"R norm should be ~{init_scale}, got {norm}"

    def test_finite_output(self):
        from fist_lora.init import gradient_projected_R, plain_svd

        model, loader = make_simple_model_and_loader()
        rank = 4

        W_q = model.query.weight.data.float()
        B, S, A = plain_svd(W_q, rank)

        frozen_BA = {"query": (B, A)}
        R_dict = gradient_projected_R(
            model, loader, frozen_BA, ["query"],
            num_samples=16, rank=rank, init_scale=0.01,
        )

        assert torch.isfinite(R_dict["query"]).all()


class TestZeroR:
    def test_shape(self):
        from fist_lora.init import zero_R
        R = zero_R(8)
        assert R.shape == (8, 8)

    def test_all_zeros(self):
        from fist_lora.init import zero_R
        R = zero_R(16)
        assert (R == 0).all()


class TestSigmaR:
    def test_shape(self):
        from fist_lora.init import sigma_R
        S = torch.tensor([3.0, 2.0, 1.5, 1.0, 0.5, 0.3, 0.1, 0.05])
        R = sigma_R(S, 4, init_scale=0.01)
        assert R.shape == (4, 4)

    def test_diagonal(self):
        from fist_lora.init import sigma_R
        S = torch.tensor([3.0, 2.0, 1.5, 1.0])
        R = sigma_R(S, 4, init_scale=0.01)
        # Should be diagonal
        off_diag = R - torch.diag(torch.diag(R))
        assert (off_diag == 0).all()

    def test_norm(self):
        from fist_lora.init import sigma_R
        S = torch.tensor([3.0, 2.0, 1.5, 1.0])
        init_scale = 0.01
        R = sigma_R(S, 4, init_scale=init_scale)
        assert abs(R.norm().item() - init_scale) < 1e-5
