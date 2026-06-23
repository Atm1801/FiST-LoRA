# Reproducibility

## Software

Exact versions are pinned in `pyproject.toml`, `requirements/lock-cuda.txt` (GPU runs) and
`requirements/lock-cpu.txt` (the full CPU environment the test suite was run with). The key versions are torch 2.4.1,
transformers 4.45.2, peft 0.13.2, datasets 2.21.0, bitsandbytes 0.44.1 and lm-eval 0.4.5. Each run records the
installed versions, git commit (and dirty flag), CUDA/cuDNN versions and GPU names in its `manifest.json`.

## Models and data (pinned revisions)

| Artifact | Source | Revision |
|---|---|---|
| RoBERTa-large | `FacebookAI/roberta-large` | `722cf37b1afa9454edce342e7895e588b6ff1d59` |
| LLaMA-2-7B (gated) | `meta-llama/Llama-2-7b-hf` | `01c7f73d771dfac7d292323805ebc428287df4f9` |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` |
| GLUE | `nyu-mll/glue` | `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` |
| CommonSense170K | LLM-Adapters GitHub, `ft-training_set/commonsense_170k.json` | commit `fe675038fb60d61b3fc03d98673d6d11d2bef4f9`, git blob `529e426b33d4cdb1d0c4a275a1a7245ad081fd91` (verified on load) |
| MetaMathQA | `meta-math/MetaMathQA` | `aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18`; 50K subset = `shuffle(seed=42)[:50000]` |
| GSM8K | `openai/gsm8k` (`main`, test, 1,319) | `740312add88f781978c0658806c59bc2815b9866` |
| MATH | `EleutherAI/hendrycks_math` (7 subjects, test, 5,000) | `21a5633873b6a120296cce3e2df9d5550074f4a3` |
| Commonsense benchmarks | lm-eval-harness 0.4.5 task definitions | pinned by the package version; task versions are stored per run |

`hendrycks/competition_math` has been removed from the Hugging Face Hub. The
EleutherAI mirror has the same 5,000 test problems split by subject.

## Seeds and randomness

- **Run seed** (42, 123, 456): python/numpy/torch/CUDA are seeded immediately before the training model is loaded
  (fixing the random head) and passed to the Trainer as `seed` and `data_seed` (data order, dropout). LoRA-XS's
  Gaussian R and LoRA-SB's estimation loader use dedicated generators seeded by the run seed.
- **Calibration seed** (`calibration.seed = 42`): used for head warm-up, its data order and the 256-example
  calibration sample. It is independent of the run seed, so one calibration serves all seeds.
  The global RNG state is restored after calibration.
- **Determinism flags** (`fist_lora/reproducibility.py`): `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN deterministic
  with benchmark off, `torch.use_deterministic_algorithms(True, warn_only=True)`, and TF32 off. Kernels with no deterministic CUDA implementation emit a warning instead of aborting a run.
  Bitwise reproducibility therefore holds only on the same hardware and software stack. On CPU, two identical
  runs give bit-identical metrics (`test_glue_run_is_deterministic_resumable_and_trains_R`).
- `group_by_length=True` (GLUE) is seeded through the Trainer.

## Caching and resumption

- The FiST calibration is cached under `cache/calibration/<task>/<key>.pt`. The key hashes the model (name,
  revision, precision, targets), the task's data fields, the calibration config, the largest rank and the package
  version. Changing any of these produces a new key; bookkeeping fields such as paths do not.
- A run is complete when `metrics.json` exists and its `config_hash` equals the hash of the current resolved
  config. `scripts/sweep.py` skips complete runs and re-runs everything else. A failed run never writes
  `metrics.json`.
- For the 7B runs, `train_metrics.json` and `adapter.pt` are written before the external evaluation.
  `scripts/evaluate.py` can redo a failed evaluation from them.

## No hidden state

Every path in a config is resolved relative to the repository root, never the working directory. The Hugging Face
cache location follows the standard `HF_HOME` variable. No experiment reads anything outside the config, the
pinned hub revisions and the cache directory.
