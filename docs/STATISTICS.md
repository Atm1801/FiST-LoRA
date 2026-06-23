# Statistical analysis

Implemented in `fist_lora/stats/` and run by `scripts/aggregate.py`.

## Unit of analysis

The experimental unit is **one trained model**: one (task, method, rank, seed) combination. The experiments use three
seeds {42, 123, 456}, so every table cell rests on n = 3 independent training runs. Examples inside a benchmark
are *not* replicates. A benchmark score is a property of one trained model, and treating its examples as
independent samples would be pseudoreplication. The pipeline never does that.

All per-seed values are kept in `summary/per_seed.csv`.

## Per-cell summaries (`aggregate.csv`)

For each (task, method, rank), over seeds:

- mean (the value shown in the result tables);
- sample standard deviation (ddof = 1);
- standard error, SD/√n;
- 95% t confidence interval, mean ± t_{0.975, n−1}·SE. At n = 3, t = 4.30, so this interval is wide by
  construction; that is the honest uncertainty from three seeds.

**GLUE value per run:** the best validation metric across epochs. The final-epoch value is stored
as well (`scripts/aggregate.py --glue-value final`). Selecting the best epoch on the reported split biases
absolute values upwards. The bias applies equally to every method, but it is still selection on the test data.

**Suite average ("Avg." column):** for each seed, the average over tasks is taken first, and the mean/SD/CI are
then computed over seeds. When every cell is complete this mean equals the average of the per-task means, and the
seed is still the replicate.

## Comparisons (`comparisons.csv`)

The planned comparisons are FiST-LoRA vs {LoRA-XS, LoRA-SB, FiST (no Fisher)} at each shared rank, and FiST-LoRA
at its largest rank vs the full-rank methods (LoRA, PiSSA, full FT).

**Pairing.** Within a task, runs with the same seed share the training-data order (Trainer `seed = data_seed`)
and the task-head initialisation. The head is created under the run seed before any adapter consumes randomness
(`test_head_initialisation_is_identical_across_methods_for_a_seed`). Differences are therefore taken per seed:
d_s = x_s − y_s.

### Per task (level `task`)

- Mean difference with a 95% t-interval over the 3 paired differences.
- Two-sided paired t-test (H0: E[d] = 0; assumes the d_s are normally distributed).
- Exact sign-flip permutation test. H0: the distribution of d is symmetric about 0; alternative two-sided;
  statistic |mean d|. **With n = 3 the smallest attainable p-value is 2/2³ = 0.25**, so no single-task comparison
  can reach conventional significance. This is reported as it is.

### Across a suite (level `suite`)

- The unit is the task (6 GLUE tasks, 8 commonsense benchmarks, 2 math benchmarks). Each task contributes its
  seed-averaged paired difference.
- Exact Wilcoxon signed-rank test (H0: task-level differences are symmetric about 0; two-sided) and a
  one-sample t-test on the task differences (Demšar, 2006).
- Tasks are a fixed, deliberately chosen set, not a random sample. The p-values describe how consistent the
  difference is across the evaluated tasks, not generalisation to unseen tasks.
- With 2 math benchmarks no test is informative (minimum Wilcoxon p = 0.5). Only the differences and intervals
  should be read.

### Multiplicity

The `holm_pvalue` column applies Holm–Bonferroni within each family:

- all task-level comparisons of the experiment;
- all suite-level comparisons of the experiment.

It is applied to the exact (permutation or Wilcoxon) p-values.

### Effect sizes

The effect size is the raw difference in metric points, with its interval. Standardised effect sizes (Cohen's d)
are not reported: with n = 3 the SD in the denominator is too unstable to be meaningful.

## What is deliberately not done

- Test-set examples are never used as replicates, including bootstrap or McNemar tests pooled across seeds.
  Per-example predictions for the math benchmarks are saved (`predictions_*.jsonl`), so test-set sampling
  uncertainty for a single model can be studied separately. It must be labelled as such.
- Calibration variability is not part of the seed spread. The FiST calibration is computed once per task and
  shared by the three seeds. LoRA-SB's gradient estimate is per seed, because the original method estimates it
  from the model being trained.
