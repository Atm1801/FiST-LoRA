"""Per-example Fisher and mean gradient via hooks == brute-force autograd."""

import pytest
import torch
from torch.utils.data import DataLoader

from conftest import (
    LLAMA_TARGETS,
    ROBERTA_TARGETS,
    PadCollator,
    cls_rows,
    lm_rows,
    pad_batch,
    tiny_llama,
    tiny_roberta,
)
from fist_lora.calibration.grad_stats import (
    GradientRecorder,
    activation_gradients,
    fisher_and_mean_gradient,
    module_groups,
    per_example_losses,
)
from fist_lora.modeling.targets import resolve_targets


def brute_force(model, rows, targets, task_type):
    """F and G from one example at a time, with W itself requiring grad (no padding at all)."""
    for m in targets.values():
        m.weight.requires_grad_(True)
    F = {n: torch.zeros_like(m.weight) for n, m in targets.items()}
    G = {n: torch.zeros_like(m.weight) for n, m in targets.items()}
    for row in rows:
        loss = per_example_losses(model, pad_batch([row], task_type), task_type).sum()
        model.zero_grad()
        loss.backward()
        for n, m in targets.items():
            F[n] += m.weight.grad**2
            G[n] += m.weight.grad
    for m in targets.values():
        m.weight.requires_grad_(False)
        m.weight.grad = None
    return {n: v / len(rows) for n, v in F.items()}, {n: v / len(rows) for n, v in G.items()}


@pytest.mark.parametrize("microbatch", [1, 4])
@pytest.mark.parametrize("regression", [False, True])
def test_seq_cls_per_example_fisher_matches_brute_force(microbatch, regression):
    model = tiny_roberta(num_labels=1 if regression else 2)
    for p in model.parameters():
        p.requires_grad_(False)
    targets = resolve_targets(model, ROBERTA_TARGETS, 8)
    rows = cls_rows(10, regression=regression)
    loader = DataLoader(rows, batch_size=microbatch, collate_fn=PadCollator("seq_cls"))
    F, G, n = fisher_and_mean_gradient(model, loader, targets, "seq_cls", torch.device("cpu"))
    F_ref, G_ref = brute_force(model, rows, targets, "seq_cls")
    assert n == 10
    for name in targets:
        assert F[name].shape == targets[name].weight.shape
        assert torch.allclose(F[name], F_ref[name], rtol=1e-4, atol=1e-10), name
        assert torch.allclose(G[name], G_ref[name], rtol=1e-4, atol=1e-8), name


def test_causal_lm_per_example_fisher_matches_brute_force():
    model = tiny_llama()
    for p in model.parameters():
        p.requires_grad_(False)
    targets = resolve_targets(model, LLAMA_TARGETS, 14)
    rows = lm_rows(6)
    loader = DataLoader(rows, batch_size=3, collate_fn=PadCollator("causal_lm"))
    F, G, _ = fisher_and_mean_gradient(model, loader, targets, "causal_lm", torch.device("cpu"))
    F_ref, G_ref = brute_force(model, rows, targets, "causal_lm")
    for name in targets:
        assert torch.allclose(F[name], F_ref[name], rtol=1e-4, atol=1e-10), name
        assert torch.allclose(G[name], G_ref[name], rtol=1e-4, atol=1e-8), name


def test_fisher_is_not_the_square_of_the_batch_gradient():
    """The empirical Fisher squares per-example gradients, not the batch-mean gradient."""
    model = tiny_roberta()
    targets = resolve_targets(model, ROBERTA_TARGETS, 8)
    rows = cls_rows(8)
    F, G, _ = fisher_and_mean_gradient(model, DataLoader(rows, batch_size=8, collate_fn=PadCollator("seq_cls")), targets, "seq_cls", torch.device("cpu"))
    name = next(iter(targets))
    assert not torch.allclose(F[name], G[name] ** 2, rtol=1e-2, atol=0.0)
    assert torch.all(F[name] >= G[name] ** 2 - 1e-12)  # Jensen: E[g^2] >= (E g)^2


def test_one_pass_equals_two_passes_and_module_groups():
    model = tiny_roberta()
    targets = resolve_targets(model, ROBERTA_TARGETS, 8)
    rows = cls_rows(6)

    def loader():
        return DataLoader(rows, batch_size=2, collate_fn=PadCollator("seq_cls"))

    F1, G1, _ = fisher_and_mean_gradient(model, loader(), targets, "seq_cls", torch.device("cpu"))
    names = sorted(targets)
    groups = list(module_groups(names, 3))
    assert [len(g) for g in groups] == [3, 3, 2]
    for group in groups:
        F2, G2, _ = fisher_and_mean_gradient(model, loader(), {k: targets[k] for k in group}, "seq_cls", torch.device("cpu"))
        for k in group:
            assert torch.equal(F1[k], F2[k]) and torch.equal(G1[k], G2[k])


def test_calibration_requires_eval_mode_and_activation_grads():
    model = tiny_roberta().train()
    targets = resolve_targets(model, ROBERTA_TARGETS, 8)
    with pytest.raises(RuntimeError, match="eval mode"):
        fisher_and_mean_gradient(model, [], targets, "seq_cls", torch.device("cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with GradientRecorder(targets, per_example=True), pytest.raises(RuntimeError, match="does not require grad"):
        model(**pad_batch(cls_rows(2), "seq_cls"))


def test_total_mode_equals_autograd_gradient_of_batch_loss():
    model = tiny_roberta()
    targets = resolve_targets(model, ROBERTA_TARGETS, 8)
    batch = pad_batch(cls_rows(5), "seq_cls")
    for p in model.parameters():
        p.requires_grad_(False)
    with activation_gradients(model), GradientRecorder(targets, per_example=False) as rec:
        model(**batch).loss.backward()
    for m in targets.values():
        m.weight.requires_grad_(True)
    model.zero_grad()
    model(**batch).loss.backward()
    for name, m in targets.items():
        assert torch.allclose(rec.grad_sum[name], m.weight.grad, rtol=1e-4, atol=1e-8)
