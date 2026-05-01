# Issue 848 Issue-791 Eval-Aligned Benchmark Rerun

## Goal

Add the missing execution surfaces for issue `#848` so the paper-facing matrix can be rerun with
the issue-791 Wave-5 PPO leader on the canonical camera-ready benchmark path.

## Problem Observed On The Original Issue Branch

Issue `#848` referenced these repository surfaces as if they already existed:

- `configs/baselines/ppo_issue_791_eval_aligned_large_capacity.yaml`
- `configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml`
- `SLURM/Auxme/issue_791_benchmark.sl`

They were absent from this checkout, even though the benchmark runner and the existing compare
config pattern already supported the issue.

## Original Branch Decision

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

## Current Mainline State After Sync

After merging `origin/main` on 2026-04-29, the issue-791 leader promotion work is already present
in mainline with a stricter, newer configuration:

- `configs/baselines/ppo_issue_791_eval_aligned_large_capacity.yaml` now resolves the promoted
  policy by `model_id` through `model/registry.yaml`, which points at the durable W&B
  `best-success:v9` artifact and carries the predictive-foresight settings used by the promoted
  PPO baseline.
- `configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml` keeps the
  issue-791 PPO leader under the `ppo` planner key, adds `workers: 1` for PPO CUDA loading, and
  uses `stop_on_failure: false` so dependency misses for other planners do not halt the whole
  publication rerun.
- `SLURM/Auxme/issue_791_benchmark.sl` remains the canonical cluster wrapper for the issue-791
  publication rerun.
- `model/registry.yaml` contains the promoted leader provenance, but the active benchmark baseline
  no longer depends on registry resolution for this PPO config.

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

Validation run after syncing with `origin/main` on 2026-04-29:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml \
  --mode preflight \
  --label issue848_preflight_after_main_sync
```

Result:

- preflight campaign:
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue848_preflight_after_main_sync_20260429_161356`
- matrix summary generated for 47 scenarios, 7 planners, differential-drive kinematics, and eval
  seeds `[111, 112, 113]`.
- AMV coverage remains `warn` under the existing `amv-paper-v1` warn-only contract.

Cluster execution wrapper:

```bash
ISSUE791_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml \
ISSUE791_BENCHMARK_LABEL=issue791-eval-aligned-leader-11724-publication \
sbatch SLURM/Auxme/issue_791_benchmark.sl
```

## Cluster Follow-up Completion

The follow-up jobs that blocked the draft PR on 2026-04-30 have completed:

- SLURM job `12208` (`robot-sf-issue791-benchmark`) completed successfully on `l40s` at
  2026-04-30 19:12:35 CEST with exit code `0:0`.
- SLURM job `12209` (`robot-sf-policy-attention-head`) completed successfully on `l40s` at
  2026-05-01 04:09:14 CEST with exit code `0:0`.

Publication rerun evidence from job `12208`:

- campaign id:
  `paper_experiment_matrix_v1_issue791-eval-aligned-leader-11724-publication_20260430_183907`
- log: `output/slurm/12208-issue791-benchmark.out`
- synced output:
  `output/slurm/issue791-benchmark-job-12208/benchmarks/publication/paper_experiment_matrix_v1_issue791-eval-aligned-leader-11724-publication_20260430_183907_publication_bundle.tar.gz`
- report root:
  `output/benchmarks/issue_791/paper_experiment_matrix_v1_issue791-eval-aligned-leader-11724-publication_20260430_183907`
- result: `7` successful runs out of `7`, `987` total episodes, `benchmark_success=true`,
  `snqi_contract_status=pass`, and `amv_coverage_status=warn`.
- PPO leader row: `141` episodes, `success_mean=0.2482`, `collisions_mean=0.0922`,
  `snqi_mean=-0.2948`, `execution_mode=native`, `readiness_tier=experimental`.

Attention-head follow-up evidence from job `12209`:

- log:
  `../origin-2026-04-28-policy_search/output/slurm/12209-issue791-attention-head.out`
- synced output:
  `../origin-2026-04-28-policy_search/output/slurm/issue791-attention-head-job-12209`
- W&B run: `https://wandb.ai/ll7/robot_sf/runs/1pcgywvr`
- best checkpoint summary:
  `success_rate=0.4857`, `collision_rate=0.5143`, `snqi=-0.9963`, selected at
  `eval_step=5242880`, with `meets_convergence=false`.

## Remaining Risks

- The issue body's held-out OOD requirement remains open; this note only restores the benchmark
  rerun surface.
- External write-ups must keep the benchmark-set / in-distribution caveat explicit.
- The publication bundle from job `12208` is still an ignored, worktree-local artifact until it is
  uploaded to a durable release/artifact store.
- The attention-head follow-up completed but did not produce a promotion-strength result.
