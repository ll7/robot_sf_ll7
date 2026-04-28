# Issue 848 Issue-791 Eval-Aligned Benchmark Rerun

## Goal

Add the missing execution surfaces for issue `#848` so the paper-facing matrix can be rerun with
the issue-791 Wave-5 PPO leader on the canonical camera-ready benchmark path.

## Problem Observed On This Branch

Issue `#848` referenced these repository surfaces as if they already existed:

- `configs/baselines/ppo_issue_791_eval_aligned_large_capacity.yaml`
- `configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml`
- `SLURM/Auxme/issue_791_benchmark.sl`

They were absent from this checkout, even though the benchmark runner and the existing compare
config pattern already supported the issue.

## Decision

Use the existing `model_id`-based PPO baseline pattern rather than hard-coding a local-only
`model_path`.

Why this matters:

- the local artifact for the issue-791 leader is not present on this laptop,
- `robot_sf.benchmark` already resolves PPO `model_id` entries through `model/registry.yaml`,
- missing local artifacts can be downloaded from W&B on the machine that executes the real run.

## Added Surfaces

- `configs/baselines/ppo_issue_791_eval_aligned_large_capacity.yaml`
- `configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml`
- `SLURM/Auxme/issue_791_benchmark.sl`
- `model/registry.yaml` entry:
  `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`

## Validation Boundary

This laptop is not the place for the full publication rerun:

- `allow_long_training_local: false`
- `allow_slurm_submission: false`

So the local validation boundary is:

1. the new compare config parses through the canonical camera-ready preflight path,
2. the matrix-definition artifacts are generated,
3. the real benchmark run is left to the cluster wrapper.

Canonical local preflight command:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml \
  --mode preflight \
  --label issue848_preflight_local
```

Cluster execution wrapper:

```bash
ISSUE791_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml \
ISSUE791_BENCHMARK_LABEL=issue791-eval-aligned-leader-11724-publication \
sbatch SLURM/Auxme/issue_791_benchmark.sl
```

## Remaining Risks

- The W&B-backed model download path still needs to be exercised on the execution machine.
- The issue body's held-out OOD requirement remains open; this note only restores the benchmark
  rerun surface.
- External write-ups must keep the benchmark-set / in-distribution caveat explicit.