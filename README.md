# FiST-LoRA: Fisher-Informed Subspace Training for Low-Rank Adaptation

A parameter-efficient fine-tuning method for large language models and vision transformers. FiST-LoRA replaces the standard SVD-based initialization in the LoRA-XS architecture with **Fisher-information-weighted SVD** for the frozen outer matrices, and uses **gradient projection** to initialize the trainable inner matrix.

## Architecture

For each target linear layer with pretrained weight `W ∈ ℝ^{d×k}`:

```
Output = W·x + (α/r) · B · R · A · x
```

| Component | Shape | Status | Source |
|-----------|-------|--------|--------|
| `B` | `(d, r)` | Frozen | Fisher-weighted SVD of W |
| `A` | `(r, k)` | Frozen | Fisher-weighted SVD of W |
| `R` | `(r, r)` | **Trainable** | Gradient-projected initialization |
| `α/r` | scalar | Fixed | Scaling factor (α=32, r=rank) |

**Trainable parameters per adapted module = r²** (e.g., 64 at rank 8 vs 16,384 for standard LoRA).

## Project Structure

```
fist-lora/
├── fist_lora/                  # Core library
│   ├── __init__.py
│   ├── config.py               # FiSTLoRAConfig dataclass
│   ├── fisher.py               # Diagonal Fisher information computation
│   ├── init.py                 # Fisher-weighted SVD + gradient projection + plain SVD
│   ├── layers.py               # FiSTLoRALinear layer
│   ├── model.py                # inject_fist_lora: replace nn.Linear with adapters
│   └── utils.py                # Param counting, warmup, calibration helpers
├── experiments/
│   ├── run_glue.py             # RoBERTa-large on GLUE (8 tasks)
│   ├── run_commonsense.py      # LLaMA-2-7B on CommonSense170K
│   ├── run_math.py             # Mistral-7B on MetaMathQA
│   ├── run_ablations.py        # All ablation variants
│   └── configs/                # YAML configs per experiment
├── baselines/
│   ├── run_lora.py             # Standard LoRA via PEFT
│   ├── run_lora_xs.py          # LoRA-XS (plain SVD, zero R)
│   └── run_lora_sb.py          # LoRA-SB (gradient-based init)
├── analysis/
│   ├── spectral_gap.py         # Per-layer spectral gap analysis
│   ├── gradient_alignment.py   # Gradient alignment tracking during training
│   ├── subspace_distance.py    # Chordal distance between subspaces
│   └── plot_results.py         # Paper figures
├── tests/
│   ├── test_fisher.py
│   ├── test_init.py
│   ├── test_layers.py
│   └── test_forward_correctness.py
├── fist_lora_poc/              # Original proof-of-concept (reference)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd FiST-LoRA

# Create a virtual environment (recommended)
conda create -n fist-lora python=3.11
conda activate fist-lora

# Install PyTorch (match your CUDA version)
# For CUDA 12.1:
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
# For CPU only:
pip install torch>=2.1.0

# Install dependencies
pip install -r requirements.txt
```

### Optional dependencies

```bash
# For experiment tracking
pip install wandb

# For commonsense evaluation
pip install lm-eval>=0.4.0

# For plotting
pip install matplotlib>=3.7.0
```

## Quick Start

### Run the test suite (verify installation)

```bash
cd FiST-LoRA
python -m pytest tests/ -v
```

All 41 tests should pass. This validates the core modules without needing GPUs or large models.

### Minimal usage example

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fist_lora.fisher import compute_diagonal_fisher
from fist_lora.init import fisher_weighted_svd, gradient_projected_R, zero_R
from fist_lora.model import inject_fist_lora, collect_plain_svd, collect_fisher_svd

# 1. Load model
model = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=2)

# 2. Compute Fisher (requires a calibration dataloader)
fisher_diags = compute_diagonal_fisher(model, calibration_loader, ["query", "key", "value"])

# 3. Compute Fisher-weighted SVD
svd_results = collect_fisher_svd(model, fisher_diags, ["query", "key", "value"], rank=8)

# 4. Compute gradient-projected R initialization
frozen_ba = {n: (B, A) for n, (B, S, A) in svd_results.items()}
R_dict = gradient_projected_R(model, calibration_loader, frozen_ba,
                               ["query", "key", "value"], rank=8, init_scale=0.01)

# 5. Inject adapters
B_dict = {n: BA[0] for n, BA in svd_results.items()}
A_dict = {n: BA[2] for n, BA in svd_results.items()}
model = inject_fist_lora(model, ["query", "key", "value"], B_dict, A_dict, R_dict,
                          alpha=32.0, rank=8)

# 6. Train with HuggingFace Trainer (only R matrices + classifier are trainable)
```

## Running Experiments

### Experiment 1: GLUE (RoBERTa-large)

This is the primary experiment. It runs 5 methods across 8 GLUE tasks with 3 seeds.

```bash
# Full run (all 8 tasks, ranks 8 and 32, seeds 42/123/456)
# Estimated time: ~24-48 hours on a single GPU
python experiments/run_glue.py

# Quick test run (1 task, 1 rank, 1 seed — ~20 min on GPU, ~2 hrs on CPU)
python experiments/run_glue.py \
    --tasks sst2 \
    --ranks 8 \
    --seeds 42 \
    --methods lora lora_xs fist_full

# Run specific tasks
python experiments/run_glue.py \
    --tasks sst2 mrpc qnli rte \
    --ranks 8 32 \
    --methods lora lora_xs fist_no_fisher fist_full pissa

# Use YAML config
python experiments/run_glue.py --config experiments/configs/glue.yaml

# Enable Weights & Biases tracking
python experiments/run_glue.py --report_to wandb
```

**Results are saved to** `results/glue/results.json` and support automatic resumption — if the run is interrupted, re-running the same command will skip completed experiments.

**Hardware requirements**: Single GPU with ≥16 GB VRAM (e.g., V100, A100, RTX 3090). RoBERTa-large is ~1.4 GB in fp32. CPU works but is very slow (~2 hours per task/method).

### Experiment 2: CommonSense Reasoning (LLaMA-2-7B)

```bash
# Full run (ranks 32/64, 3 seeds)
# Estimated time: ~48-72 hours on a single A100
python experiments/run_commonsense.py

# Quick test (1 rank, 1 seed, skip evaluation)
python experiments/run_commonsense.py \
    --ranks 32 \
    --seeds 42 \
    --methods lora_xs fist_full \
    --skip_eval

# With evaluation on all 8 commonsense tasks
python experiments/run_commonsense.py \
    --ranks 32 \
    --seeds 42 \
    --methods fist_full
```

**Hardware requirements**: Single GPU with ≥24 GB VRAM. Uses 4-bit quantization (bitsandbytes) to fit LLaMA-2-7B.

**Data setup**: Download CommonSense170K from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters) and place at `data/commonsense_170k.json`.

### Experiment 3: Math Reasoning (Mistral-7B)

```bash
# Full run (ranks 32/64/96, 3 seeds)
python experiments/run_math.py

# Quick test
python experiments/run_math.py \
    --ranks 32 \
    --seeds 42 \
    --methods fist_full lora \
    --max_train_samples 5000 \
    --skip_eval
```

**Hardware requirements**: Same as CommonSense (single A100 or equivalent).

### Experiment 4: Ablation Study

```bash
# Full ablation on GLUE (all 8 variants × 4 tasks × 2 ranks × 3 seeds)
python experiments/run_ablations.py

# Quick subset
python experiments/run_ablations.py \
    --tasks sst2 mrpc \
    --ranks 8 \
    --seeds 42 \
    --variants fist_full no_fisher no_grad lora_xs
```

**Ablation variants tested**:

| ID | Outer Init | Inner Init | Tests |
|----|-----------|------------|-------|
| `fist_full` | Fisher-SVD | Gradient (scale=0.01) | Full method |
| `no_fisher` | Plain SVD | Gradient (scale=0.01) | Fisher contribution |
| `no_grad` | Fisher-SVD | Zero | Gradient init contribution |
| `lora_xs` | Plain SVD | Zero | LoRA-XS baseline |
| `sigma_init` | Fisher-SVD | diag(σ) scaled | Alternative init |
| `scale_sweep_001` | Fisher-SVD | Gradient (scale=0.001) | Scale sensitivity |
| `scale_sweep_01` | Fisher-SVD | Gradient (scale=0.01) | Scale sensitivity (default) |
| `scale_sweep_1` | Fisher-SVD | Gradient (scale=0.1) | Scale sensitivity |

### Running Individual Baselines

```bash
# Standard LoRA
python baselines/run_lora.py --task sst2 --rank 8 --seed 42

# LoRA-XS
python baselines/run_lora_xs.py --task sst2 --rank 8 --seed 42

# LoRA-SB
python baselines/run_lora_sb.py --task sst2 --rank 8 --seed 42
```

## Running Analysis

After experiments complete, generate analysis plots and paper figures:

```bash
# Spectral gap analysis (Fisher-SVD vs plain SVD)
python analysis/spectral_gap.py --model roberta-large --task sst2 --rank 8

# Gradient alignment during training
python analysis/gradient_alignment.py --model roberta-large --task sst2 --rank 8

# Subspace distance heatmaps
python analysis/subspace_distance.py --model roberta-large --task sst2 --rank 8

# Generate paper figures from all results
python analysis/plot_results.py --results_dir results/ --output_dir results/figures/
```

Outputs are saved to `results/analysis/` and `results/figures/`.

## Key Implementation Details

### PoC-Validated Design Decisions

These choices were validated across 3 iterative debugging rounds on RoBERTa-large + 4 GLUE tasks:

1. **Fisher clipping (95th percentile)**: Without this, extreme Fisher values dominate the SVD and select a degenerate subspace. The clipping + mean normalization is in `fist_lora/init.py:fisher_weighted_svd()`.

2. **R init_scale=0.01**: The gradient-projected R direction is preserved but magnitude is controlled to `||R||_F = 0.01`. This gives `||ΔW|| ≈ (α/r) × 0.01 ≈ 0.04` (a gentle nudge). The previous formula produced `||ΔW|| ≈ 45` which destroyed pretrained representations.

3. **Head warmup for calibration only**: The classifier head is warmed up for 100 steps before computing Fisher/gradients, but this warmed state is **never loaded into training models**. Each training model starts with a fresh random head. Loading warmed state causes gradient conflicts and training collapse.

4. **No pooler unfreezing**: The pooler is pretrained (not random) and including it adds ~786K params that inflate the count unfairly vs PEFT LoRA.

5. **Higher LR for FiST-LoRA**: With only r² trainable adapter params, FiST-LoRA needs a higher learning rate (1e-3 for RoBERTa, 1e-4 for 7B LLMs) compared to standard LoRA (2e-4 / 2e-5).

### Learning Rates

| Method | RoBERTa-large | 7B LLMs |
|--------|--------------|---------|
| LoRA / PiSSA | 2e-4 | 2e-5 |
| LoRA-XS / FiST-LoRA | 1e-3 | 1e-4 |

If results look weak, sweep: `{5e-4, 1e-3, 2e-3, 5e-3}` for RoBERTa; `{5e-5, 1e-4, 2e-4}` for 7B.

## Methods Compared

| Method | Outer Matrices | Inner Matrix | Trainable Params/Module |
|--------|---------------|-------------|------------------------|
| LoRA | Random A, B | Both A, B trainable | 2r(d+k) |
| PiSSA | SVD of W | Both trainable (principal) | 2r(d+k) |
| LoRA-XS | Plain SVD of W | Zero-init R (r×r) | r² |
| LoRA-SB | Gradient SVD | Gradient-init R | r² |
| **FiST-LoRA** | **Fisher-weighted SVD** | **Gradient-projected R** | **r²** |

## Interpreting Results

The key claims to evaluate:

1. **fist_full > lora_xs consistently** → Fisher-weighted SVD selects a better subspace than plain SVD
2. **fist_no_fisher > lora_xs** → Gradient-projected R provides a useful warm start
3. **fist_full within 2-3% of LoRA** → Competitive at dramatically fewer parameters (~100-500× fewer adapter params)

If fist_full does NOT beat lora_xs: the Fisher contribution is weak, and the paper should focus on gradient-projected initialization.

## Citation

```bibtex
@article{fistlora2024,
  title={FiST-LoRA: Fisher-Informed Subspace Training for Low-Rank Adaptation},
  author={...},
  year={2024}
}
```
