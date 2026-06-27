# FiST-LoRA: Fisher-Informed Subspace Training

FiST-LoRA is an extremely parameter-efficient adapter in the frozen-outer, r²-budget family (like LoRA-XS and
LoRA-SB):

```
h = W0 x + (α/r) · B (R (A x))        B ∈ R^{d×r}, A ∈ R^{r×k} frozen;  R ∈ R^{r×r} trained (r² parameters)
```

It differs from its relatives in how the frozen subspace and the starting point are chosen:

1. **Fisher-weighted SVD.** B and A are the top-r singular vectors of `√(F̄ + ε) ⊙ W0`, where F̄ is the clipped,
   normalised diagonal (empirical) Fisher, estimated on 256 calibration examples.
2. **Gradient-projected initialisation.** `R_init = γ · BᵀGAᵀ / ‖BᵀGAᵀ‖_F`, where G is the mean calibration
   gradient and γ = 0.01.

This repository implements FiST-LoRA and the baselines it is compared against (full fine-tuning, LoRA, PiSSA,
LoRA-XS, LoRA-SB). It covers three experimental suites: GLUE on RoBERTa-large, CommonSense170K on LLaMA-2-7B and
MetaMathQA on Mistral-7B. Component ablations, result tables, training-dynamics figures and a seed-level
statistical analysis are included.

## Contents

- [Installation](#installation)
- [Repository layout](#repository-layout)
- [Models and datasets](#models-and-datasets)
- [Methods](#methods)
- [Running experiments](#running-experiments)
- [Experiment suites](#experiment-suites)
- [Statistical analysis](#statistical-analysis)
- [Expected outputs](#expected-outputs)
- [Reproducibility](#reproducibility)
- [Tests](#tests)

## Installation

Python 3.10 or 3.11.

```bash
git clone git@github.com:Atm1801/FiST-LoRA.git && cd FiST-LoRA
python -m venv .venv && source .venv/bin/activate

# GPU machine (full experiments): CUDA 12.1 wheels, bitsandbytes, lm-eval
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/lock-cuda.txt
pip install -e . --no-deps

# CPU only (development, tests, smoke runs)
pip install -e ".[dev,eval]"
```

Every package version is pinned (`pyproject.toml`; `requirements/lock-cpu.txt` is the exact environment the
tests were run in). LLaMA-2-7B is gated: accept the licence on the Hugging Face Hub, then run
`huggingface-cli login`.

## Repository layout

```
fist_lora/
  config/         typed experiment schema, YAML loading (_base_ composition, --set overrides)
  modeling/       model/tokenizer loading at pinned revisions, target regexes, NF4 helpers
  adapters/       FrozenOuterLinear, injection/freezing, PEFT LoRA & PiSSA
  calibration/    head warm-up, per-example Fisher & mean gradient, clipping/normalisation, SVD,
                  inner-matrix initialisations, LoRA-SB update estimation, calibration cache
  methods/        registry: every method = (outer subspace, inner init, scale, calibration)
  data/           GLUE, CommonSense170K, MetaMathQA, calibration sampling, collators
  training/       Trainer set-up, single-run orchestration, adapter restore
  evaluation/     GLUE metrics + best epoch, lm-eval wrapper, GSM8K/MATH generation + extractor
  stats/          per-seed records, seed aggregation, paired comparisons
  reporting/      result tables, training-dynamics figure
  params.py       analytic parameter counts and adapter storage
scripts/          command-line entry points (below)
configs/          models/, tasks/, experiments/ (one YAML per suite / ablation / smoke test)
docs/             METHODS.md, STATISTICS.md, REPRODUCIBILITY.md
tests/            unit, integration (tiny models) and GPU tests
```

## Models and datasets

| | Source | Pinned revision |
|---|---|---|
| RoBERTa-large (355M) | `FacebookAI/roberta-large` | `722cf37b…` |
| LLaMA-2-7B | `meta-llama/Llama-2-7b-hf` (gated) | `01c7f73d…` |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | `27d67f1b…` |
| GLUE: CoLA, RTE, MRPC, STS-B, QNLI, SST-2 | `nyu-mll/glue` | `bcdcba79…` |
| CommonSense170K | LLM-Adapters `commonsense_170k.json` @ `fe675038` | downloaded automatically to `data/`, git-blob checksum verified |
| MetaMathQA (50K seeded subset) | `meta-math/MetaMathQA` | `aa4f34d3…` |
| GSM8K (test, 1,319) | `openai/gsm8k` | `740312ad…` |
| MATH (test, 5,000) | `EleutherAI/hendrycks_math` (mirror; `hendrycks/competition_math` was removed from the Hub) | `21a56338…` |
| BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e/c, OBQA | lm-eval-harness 0.4.5 task definitions | package version |

Full SHAs are in the configs and in `docs/REPRODUCIBILITY.md`. Everything is downloaded on first use; there is no
manual data preparation step.

## Methods

| Name in configs | Method | Trainable per module |
|---|---|---|
| `full_ft` | full fine-tuning (GLUE only) | all |
| `lora` | LoRA via PEFT (B = 0) | r(d+k) |
| `pissa` | PiSSA via PEFT (bf16 backbones only) | r(d+k) |
| `lora_xs` | LoRA-XS (Σ in the input factor, R ~ N(0, 1e-10)) | r² |
| `lora_sb` | LoRA-SB (SVD of −η·sign(Σ∇), R = S_r, s = 1) | r² |
| `fist_no_fisher` | plain-SVD outer + gradient-projected R | r² |
| `fist` | **FiST-LoRA**: Fisher-weighted SVD outer + gradient-projected R | r² |
| `svd_zero`, `fisher_zero`, `svd_sigma`, `fisher_sigma`, `fist_gamma*` | component ablations | r² |

In the GLUE experiments the classification head is trained in every method; in the 7B experiments the pretrained
`lm_head` is frozen in every method. "# Params" never counts the head. The full specification, the calibration
procedure and per-baseline fidelity checklists are in [`docs/METHODS.md`](docs/METHODS.md).

### Baselines

LoRA-XS and LoRA-SB follow their original papers and official implementations:

- **LoRA-XS:** singular values absorbed into the frozen input-side factor, R ~ N(0, 10⁻¹⁰).
- **LoRA-SB:** the first update is estimated from 2 / 170 / 50 examples (GLUE / commonsense / math) as
  −η_eff·sign(Σ∇W L), then R_init = S_r with scale 1.

The ablation `svd_zero` (plain-SVD subspace, zero R) differs from `fist_no_fisher` only in the inner
initialisation. The chain `svd_zero → fist_no_fisher → fist` therefore separates the two FiST components.

## Running experiments

Every command takes `--config` (an experiment YAML) and any number of `--set key.path=value` overrides.

**Single run** (one task × method × rank × seed; calibrates on demand):

```bash
python scripts/train.py --config configs/experiments/glue.yaml --task rte --method fist --rank 8 --seed 42
```

**Sweep** (all or a subset of runs; resumable: completed runs are skipped, failed runs are retried):

```bash
python scripts/sweep.py --config configs/experiments/glue.yaml --dry-run   # list runs
python scripts/sweep.py --config configs/experiments/glue.yaml --tasks rte mrpc --methods fist lora_xs
```

**Calibration only** (computes and caches the FiST calibration; prints Fisher diagnostics):

```bash
python scripts/calibrate.py --config configs/experiments/glue.yaml --tasks rte
```

**Evaluation.** GLUE is evaluated on the validation split after every epoch during training. The 7B runs are
evaluated right after training: lm-eval-harness for commonsense, greedy generation plus exact match for math. If
that evaluation fails, redo it from the saved adapter:

```bash
python scripts/evaluate.py --config configs/experiments/math.yaml --task metamathqa50k --method fist --rank 8 --seed 42
```

**CPU smoke tests** (tiny random models, synthetic data, a few minutes; the numbers are meaningless):

```bash
python scripts/sweep.py --config configs/experiments/smoke_glue.yaml
python scripts/sweep.py --config configs/experiments/smoke_causal.yaml
```

## Experiment suites

| Suite | Commands |
|---|---|
| **GLUE** (RoBERTa-large, 6 tasks; 252 runs) | `python scripts/sweep.py --config configs/experiments/glue.yaml`<br>`python scripts/aggregate.py --config configs/experiments/glue.yaml`<br>`python scripts/make_tables.py --config configs/experiments/glue.yaml` |
| GLUE training dynamics (r = 8) | after the GLUE sweep: `python scripts/make_figures.py --config configs/experiments/glue.yaml` |
| **Commonsense** (LLaMA-2-7B, 8 benchmarks; 39 runs) | the same three commands with `configs/experiments/commonsense.yaml` |
| **Math** (Mistral-7B, GSM8K + MATH; 39 runs) | the same three commands with `configs/experiments/math.yaml` |
| **Ablations** | `ablation_glue.yaml`, `ablation_commonsense.yaml` (sweep → aggregate → make_tables) |
| PiSSA (GLUE) | `pissa_glue.yaml` |
| Parameter counts, adapter storage, initial perturbation | `python scripts/param_tables.py` (no training) |
| Subspace overlap between methods | `python scripts/analyze_subspaces.py --config configs/experiments/glue.yaml --task rte --rank 8` |

Runs can be split across machines by passing `--tasks`, `--methods`, `--ranks` or `--seeds` to `sweep.py`. Each
run writes into its own directory, and `aggregate.py` reads whatever has been completed.

## Statistical analysis

The unit of analysis is one trained model (task × method × rank × seed), with 3 seeds per cell.
`scripts/aggregate.py` writes:

- per-seed values;
- mean, SD, SEM and 95% t-intervals over seeds;
- paired comparisons (FiST-LoRA vs each baseline).

The comparisons use seed-paired differences per task (paired t-test and exact sign-flip permutation test; with 3
seeds the smallest attainable p-value is 0.25). Across a suite, the tasks are the units (exact Wilcoxon
signed-rank test), with Holm correction. Benchmark examples are never treated as replicates. Details and
assumptions are in [`docs/STATISTICS.md`](docs/STATISTICS.md).

The GLUE value of a run is its best validation metric across epochs (`--glue-value final` gives the last epoch).

## Expected outputs

```
results/<experiment>/
  runs/<task>/<method>/<r{rank}|full>/seed<seed>/
    config.yaml            fully resolved configuration of the run
    metrics.json           results (+ config hash); written last: its presence marks completion
    log_history.jsonl      Trainer log: training loss per logging step, validation metric per epoch
    manifest.json          package versions, git commit, GPU, model/dataset revisions
    adapter.pt             trained adapter (B, A, R, scale / LoRA factors, and head)
    train_metrics.json     7B only: training results saved before evaluation
    predictions_*.jsonl    math only: per-problem generations, extracted answers, correctness
  summary/per_seed.csv, aggregate.csv, comparisons.csv
  tables/<experiment>.md, <experiment>.tex
  figures/training_dynamics.pdf, .png         (glue)
  analysis/subspaces_<task>_r<rank>.csv       (analyze_subspaces.py)
cache/calibration/<task>/<key>.pt             FiST calibration, shared by all ranks and seeds
results/param_tables.md
```

## Reproducibility

- **Seeds.** Runs use {42, 123, 456}. The run seed fixes the head initialisation, data order, dropout and the
  LoRA-XS / LoRA-SB randomness. Calibration uses its own seed (42) and is computed once per task, then shared
  across seeds.
- **Determinism.** Deterministic cuDNN/cuBLAS settings are applied and TF32 is off. Two identical CPU runs
  produce bit-identical metrics (tested). GPU runs are deterministic on the same hardware and software stack,
  apart from warnings from kernels that have no deterministic implementation.
- **Pinning.** Models, datasets, CommonSense170K (checksum-verified) and packages are pinned. Each run records its
  environment in `manifest.json`.
- **No hidden state.** Paths resolve against the repository root, calibration outputs are cached under keys that
  hash every relevant setting, and completed runs are identified by a config hash.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Tests

```bash
pytest                         # offline unit + integration tests (CPU, ~2 s)
pytest -m network              # end-to-end runs on tiny Hub models, restore, lm-eval wrapper
pytest -m gpu                  # NF4 calibration/training path (needs CUDA + bitsandbytes)
ruff check .
```

The tests cover:

- **Method maths:** closed forms for clipping, normalisation, Fisher weighting and gradient projection;
  per-example gradients against brute-force autograd; the exact initial-perturbation norm (α/r)γ; SVD sign
  invariance.
- **Shapes and targets:** shapes for square and non-square matrices, exact target modules.
- **Parameter counts:** L·M·r² and r(d+k) against real modules, including a GQA model.
- **Trainable sets and gradient flow:** under gradient checkpointing, including the silent zero-gradient failure
  that reentrant checkpointing causes with a frozen backbone.
- **Baselines:** LoRA-XS / LoRA-SB definitions, PEFT LoRA/PiSSA initialisation.
- **Metrics and statistics:** metrics, best-epoch selection, the GSM8K/MATH answer extractor, statistics against
  scipy.
- **Configs:** the experiment grids.
- **End to end:** run determinism and resumption, and exact adapter restoration.
