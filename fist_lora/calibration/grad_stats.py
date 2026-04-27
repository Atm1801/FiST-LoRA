"""Gradients of the loss with respect to (possibly quantised) linear weights, via hooks.

For a linear layer y = x W^T + b applied at token positions t, the gradient of a scalar
loss with respect to W is sum_t delta_t x_t^T with delta_t = dL/dy_t.  Capturing x in a
forward hook and delta in a tensor hook on y gives this gradient without W itself
requiring grad, which is what makes the computation possible for NF4 layers (whose
packed uint8 storage cannot carry a gradient).  The result is the gradient with respect
to the *effective* (dequantised) weight used in the forward pass.

Two accumulation modes:

* ``per_example`` (FiST Fisher and mean gradient): the per-example gradients
  g_i = sum_t delta_{i,t} x_{i,t}^T are formed explicitly (einsum over the micro-batch),
  and both sum_i g_i^2 and sum_i g_i are accumulated.  Backpropagating the *sum* of the
  per-example losses makes delta_{i,.} depend on example i only, so g_i is exactly
  dL(x_i, y_i)/dW.  Padding positions receive no gradient (they are masked out of the
  attention of real tokens and out of the loss), so batching does not change g_i.
* ``total``: only sum_t,b delta x^T of whatever scalar was backpropagated (LoRA-SB).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientRecorder:
    def __init__(self, modules: dict[str, nn.Module], per_example: bool) -> None:
        self.modules = modules
        self.per_example = per_example
        self.grad_sum: dict[str, torch.Tensor] = {}
        self.sq_sum: dict[str, torch.Tensor] = {}
        self._handles: list = []

    def __enter__(self) -> GradientRecorder:
        for name, module in self.modules.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def forward_hook(module, inputs, output):
            if not output.requires_grad:
                raise RuntimeError(
                    f"output of {name} does not require grad; enable activation gradients first"
                )
            x = inputs[0].detach()
            output.register_hook(lambda delta: self._accumulate(name, x, delta))

        return forward_hook

    @torch.no_grad()
    def _accumulate(self, name: str, x: torch.Tensor, delta: torch.Tensor) -> None:
        x, delta = x.float(), delta.float()
        if x.dim() == 2:  # (b, k) inputs without a sequence axis
            x, delta = x.unsqueeze(1), delta.unsqueeze(1)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        delta = delta.reshape(delta.shape[0], -1, delta.shape[-1])
        if self.per_example:
            g = torch.einsum("btd,btk->bdk", delta, x)  # (b, d, k) per-example gradients
            _add(self.grad_sum, name, g.sum(0))
            _add(self.sq_sum, name, g.pow(2).sum(0))
        else:
            _add(self.grad_sum, name, torch.einsum("btd,btk->dk", delta, x))


def _add(store: dict[str, torch.Tensor], name: str, value: torch.Tensor) -> None:
    if name in store:
        store[name] += value
    else:
        store[name] = value.clone()


@contextmanager
def activation_gradients(model: nn.Module):
    """Make the embedding output require grad so a fully frozen model still builds a graph.

    Equivalent to ``PreTrainedModel.enable_input_require_grads`` but removable.
    """

    def hook(module, inputs, output):
        output.requires_grad_(True)

    handle = model.get_input_embeddings().register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def per_example_losses(model: nn.Module, batch: dict[str, torch.Tensor], task_type: str) -> torch.Tensor:
    """L(x_i, y_i) for every example of a batch (shape (b,)).

    * Sequence classification: cross-entropy, or squared error when the model has a
      single output (STS-B regression), matching the Hugging Face loss definitions.
    * Causal LM: token-mean negative log-likelihood over the example's own label tokens
      (labels == -100 are ignored), i.e. the loss the example would have on its own.
    """
    labels = batch["labels"]
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    logits = model(**inputs).logits.float()
    if task_type == "seq_cls":
        if logits.shape[-1] == 1:
            return F.mse_loss(logits.squeeze(-1), labels.float(), reduction="none")
        return F.cross_entropy(logits, labels, reduction="none")
    if task_type == "causal_lm":
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        tok = F.cross_entropy(
            shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none"
        )
        mask = (shift_labels != -100).float()
        counts = mask.sum(1)
        if (counts == 0).any():
            raise ValueError("an example has no label tokens; its loss is undefined")
        return (tok * mask).sum(1) / counts
    raise ValueError(f"unknown task_type {task_type!r}")


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def fisher_and_mean_gradient(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    modules: dict[str, nn.Module],
    task_type: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    """One pass over the calibration set: F = (1/N) sum_i g_i^2 (diagonal empirical Fisher), G = (1/N) sum_i g_i.

    The model must be in eval mode, so the pass is deterministic and computing F and G in
    one pass equals two separate passes (one for F, one for G) over the same examples.
    """
    if model.training:
        raise RuntimeError("FiST calibration statistics are computed in eval mode")
    n = 0
    with activation_gradients(model), GradientRecorder(modules, per_example=True) as rec:
        for batch in batches:
            batch = to_device(batch, device)
            losses = per_example_losses(model, batch, task_type)
            losses.sum().backward()
            model.zero_grad(set_to_none=True)
            n += losses.shape[0]
    if n == 0:
        raise RuntimeError("calibration saw zero examples")
    missing = set(modules) - set(rec.grad_sum)
    if missing:
        raise RuntimeError(f"no gradient reached modules {sorted(missing)[:4]}")
    fisher = {k: v / n for k, v in rec.sq_sum.items()}
    grad = {k: v / n for k, v in rec.grad_sum.items()}
    return fisher, grad, n


def module_groups(names: list[str], per_pass: int | None) -> Iterator[list[str]]:
    """Split target modules into groups processed in separate passes (memory bound)."""
    if not per_pass or per_pass >= len(names):
        yield list(names)
        return
    for i in range(0, len(names), per_pass):
        yield names[i : i + per_pass]


@contextmanager
def replayable_rng():
    """Restore the RNG state on entry at every ``replay()`` so repeated passes see identical
    dropout masks / sampling (used when statistics are split over several passes)."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def replay() -> None:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)

    yield replay


BatchFactory = Callable[[], Iterable[dict[str, torch.Tensor]]]
