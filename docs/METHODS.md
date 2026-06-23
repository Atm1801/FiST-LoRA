# Methods

All adapted layers compute

```
h = W0 x + s · B (R (A x))          B ∈ R^{d×r}, A ∈ R^{r×k}, R ∈ R^{r×r}
```

for the frozen-outer (r²-budget) methods, or `h = W0 x + s · B A x` with trainable `B, A` for LoRA and PiSSA.
`W0` is the weight as stored in `nn.Linear` (shape `d × k`, `d = out_features`); for 4-bit backbones it is the
dequantised NF4 weight actually used in the forward pass. Code: `fist_lora/methods/registry.py`.

| Method | Frozen outer (B, A) | Inner init R | Scale s | Trainable | Data used |
|---|---|---|---|---|---|
| Full FT | – | – | – | all weights | – |
| LoRA | – | B = 0, A ~ kaiming-uniform(√5) | α/r | A, B (+ head) | – |
| PiSSA | – | B = U_r√S_r/√s, A = √S_r V_rᵀ/√s; W0 ← W0 − sBA | α/r | A, B (+ head) | – |
| LoRA-XS | B = U_r, A = S_r V_rᵀ from SVD(W0) | N(0, (1e-5)²) | α/r | R (+ head) | – |
| LoRA-SB | B = U_r, A = V_rᵀ from SVD(ΔW), ΔW = −η_eff·sign(Σ∇W L) | S_r (of ΔW) | 1 | R (+ head) | n examples, per seed |
| FiST (no Fisher) | B = U_r, A = V_rᵀ from SVD(W0) | γ · R_proj/‖R_proj‖_F | α/r | R (+ head) | 256 calibration examples |
| FiST-LoRA | B = Ũ_r, A = Ṽ_rᵀ from SVD(√(F̄+ε) ⊙ W0) | γ · R_proj/‖R_proj‖_F | α/r | R (+ head) | 256 calibration examples |

`(+ head)`: the task head is trained for the GLUE (sequence-classification) experiments in every method, and frozen
for the 7B causal-LM experiments in every method (the pretrained `lm_head`; see "Task head" below).

## FiST-LoRA

Calibration is computed **once per task** and shared by all ranks and seeds (`fist_lora/calibration/fist.py`):

1. **Head warm-up** (`calibration/warmup.py`). A dedicated model is loaded with seed `calibration.seed` (42). Only
   its head is trained, for 100 AdamW steps at lr 1e-3, batch 32 (RoBERTa) or 4 (7B), with the backbone frozen.
   This model is used only for calibration and is then deleted. Training runs load a fresh model.
2. **Calibration set.** N = 256 examples are drawn from the training set with a seeded uniform sample
   (`data/calibration.py`).
3. **Fisher and gradient** (`calibration/grad_stats.py`). One eval-mode pass accumulates, for every target
   module, the per-example weight gradients g_i = ∂L(x_i, y_i)/∂W:
   - `F = (1/N) Σ g_i ⊙ g_i` (diagonal empirical Fisher)
   - `G = (1/N) Σ g_i` (mean gradient)

   g_i is built from forward/backward hooks (`Σ_t δ_t x_tᵀ`), so W itself never needs `requires_grad`. This is
   what makes NF4 layers possible. Backpropagating the sum of per-example losses keeps the g_i separable. In eval
   mode one pass is identical to two separate passes (one for F, one for G) over the same examples.
4. **Clip and normalise** (`calibration/fisher.py`): `q95 = Quantile(F, 0.95)`,
   `F̄ = min(F, q95) / (mean(min(F, q95)) + ε)`, ε = 1e-8. The statistics are global over the matrix. The quantile
   uses exact linear interpolation (identical to `torch.quantile`), computed with `kthvalue` because
   `torch.quantile` rejects the > 2²⁴-element 7B matrices.
5. **Weighted SVD** (`calibration/subspace.py`): `W̃ = √(F̄ + ε) ⊙ W0`, exact SVD in fp32, `B = Ũ_r`,
   `A = Ṽ_rᵀ`.
6. **Gradient projection** (`calibration/inner_init.py`): `R_proj = Bᵀ G Aᵀ`, `R_init = γ R_proj/‖R_proj‖_F`,
   γ = 0.01. A zero projection raises an error; there is no silent fallback. Because `Bᵀ G Aᵀ` for rank r is the
   leading r × r block of the same product at the largest rank, one calibration serves r ∈ {8, 16, 24}.

Since B has orthonormal columns and A orthonormal rows, the initial update has `‖s B R_init A‖_F = (α/r)·γ`
exactly (tested). The gradient only sets the direction of R_init.

**Per-example loss L(x, y).** For classification this is the cross-entropy of one example, or the squared error
for STS-B. For causal LMs it is the mean token NLL over that example's label tokens.

**Gradient sign.** R_init projects +G, the loss gradient itself; its magnitude is at most (α/r)γ ≤ 0.04.

**Scale of the Fisher estimate.** When the mean clipped Fisher is not much larger than ε (small models or very
small gradients), ε dominates the normaliser and F̄ is not centred at one. The relative weights stay ∝ √F unless
F < ~1e-16. Calibration logs this condition per module (`modules_eps_not_negligible`, `clipped_mean`).

## Baselines: fidelity to the original methods

Each check below is marked ✓ (implemented as in the original and covered by a test) or Δ (a documented deviation).

### LoRA (Hu et al., 2022), via Hugging Face PEFT 0.13.2

- ✓ `LoraConfig(init_lora_weights=True)`: A kaiming-uniform (a = √5), B = 0, so the model is unchanged at
  initialisation (`test_peft_lora_init_and_trainable_head`).
- ✓ Scale α/r; dropout 0; no bias training; the same target modules as every other method.
- ✓ The head is trained through `modules_to_save=["classifier"]`. A test checks that the trained copy is the one
  used in forward.
- ✓ Parameter count r(d + k) per module (`test_lora_counts`, `test_peft_lora_init_and_trainable_head`).

### PiSSA (Meng et al., 2024), via PEFT `init_lora_weights="pissa"` (the authors' upstreamed implementation)

- ✓ Exact SVD, principal components in (B, A), residual W0 − sBA as the frozen base. The output is unchanged at
  initialisation (`test_pissa_preserves_function_at_init`).
- Δ An NF4 backbone is **not supported**: it raises `NotImplementedError`. QPiSSA needs the full-precision SVD and
  a re-quantised residual.

### LoRA-XS (Bałazy et al., 2024; github.com/MohammadrezaBanaei/LoRA-XS @ e50b1a8)

- ✓ Truncated SVD of W0, with the singular values absorbed into the **input-side** frozen factor. The official
  code runs SVD on Wᵀ and sets `lora_A = (U′S)ᵀ`. In the `h = W0 x` convention this is B = U_r, A = S_r V_rᵀ
  (LoRA-XS's row-vector notation "A = U_r Σ_r"). BA is the best rank-r approximation of W0
  (`test_lora_xs_original_absorbs_sigma_into_input_factor`).
- ✓ R ~ N(0, σ²), σ = 1e-5 (`init_module_weights(..., sigma=0.00001)`), seeded by the run seed.
- ✓ Only R (+ head) trainable; r² parameters per module.
- Δ The official code uses scikit-learn's randomized `TruncatedSVD(n_iter=10)`. Here an exact SVD is used for
  every method, a numerical-only difference.

### LoRA-SB (Ponkshe et al., 2024; github.com/CERT-Lab/lora-sb @ 4feb81c)

- ✓ ΔW = −η_eff · sign(Σ_batches ∇_W (L_batch / n_batch)) over the first `num_samples` examples of a shuffled
  training loader with batch size `estimation_batch_size` (`utils/gradient_utils.py`). The settings are: GLUE
  2 / 128 (`train_glue.py`), commonsense 170 / 10 (`train_cr.py`), math 50 / 3 (`train_arithmetic.py`).
- ✓ η_eff = lr / (warmup_ratio · ⌈N_train / estimation_batch_size⌉ · epochs), i.e. the learning rate of the
  first warm-up step (`effective_lr`).
- ✓ The model is in train mode and is the model that will be trained. Its head is the seed's fresh head with no
  warm-up, so the estimate is per seed.
- ✓ SVD(ΔW): B = U_r, A = V_rᵀ, R_init = S_r / s with **s = 1** (the scripts set `lora_alpha = lora_r`). B and A
  are frozen (`test_lora_sb_estimate_matches_definition`).
- Δ The official code uses `torch.svd_lowrank(niter=10)` and casts the factors to bf16. Here an exact fp32 SVD is
  used.
- Δ The warm-up ratio in η_eff is this protocol's (0.06), which the 7B scripts of the original replace with 0.02.

## Ablations

| Name | Outer | Inner |
|---|---|---|
| `svd_zero` | plain SVD (U_r, V_rᵀ) | 0 |
| `fisher_zero` | Fisher-weighted SVD | 0 |
| `svd_sigma` | plain SVD | γ·diag(S_r)/‖·‖ |
| `fisher_sigma` | Fisher-weighted SVD | γ·diag(S_r)/‖·‖ |
| `fist_gamma*` | Fisher-weighted SVD | gradient projection with γ ∈ {0.001, 0.1} |

`svd_zero → fist_no_fisher` changes only the inner initialisation, and `fist_no_fisher → fist` changes only the
subspace. Together the two steps separate the contributions of the FiST components.

## Task head

- **GLUE:** the RoBERTa classification head is trained in every method and excluded from "# Params". FiST warms a
  head on the calibration model only. Every training run starts from a fresh random head, initialised under the
  run seed before any adapter randomness, so the head is identical across methods at a given seed.
- **7B models:** the "task head" is the pretrained `lm_head`. It is warmed on the discarded calibration copy for
  calibration only. Training reloads the pretrained `lm_head` and keeps it **frozen in every method**, so that
  every method adapts the same set of weights.

## Training protocol

AdamW (Hugging Face Trainer, β = (0.9, 0.999), ε = 1e-8), weight decay 0, linear schedule, warm-up ratio 0.06.
GLUE: effective batch 128, bf16 autocast, per-task epochs and max length from `configs/tasks/glue/`, evaluation
after every epoch. 7B: NF4 + double quantisation + bf16 compute, non-reentrant gradient checkpointing, batch 4 × 16
(commonsense) or 4 × 32 (math). Learning rates: 4e-4 (LoRA/PiSSA) vs 1e-3 (r² methods) on GLUE, 2e-5 vs 1e-4 on 7B;
full fine-tuning uses 1e-5.
