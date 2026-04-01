# SLURM Submission Workflow

[← Back to Documentation Index](../README.md)

Use the shared wrapper below for new batch jobs so the requested wall time tracks the
current partition and QoS policy instead of relying on stale hardcoded `#SBATCH --time`
lines.

## Default workflow

```bash
scripts/dev/sbatch_use_max_time.sh SLURM/Auxme/auxme_gpu.sl
```

The wrapper:

- reads `#SBATCH --partition` and `#SBATCH --qos` from the target script,
- queries Slurm for the partition `MaxTime`,
- queries QoS `MaxWall` when that metadata is available,
- uses the effective maximum of the selected profile as the default `--time`, and
- passes that value to `sbatch`, which overrides the script-local time directive.

This keeps new submissions aligned with the live cluster policy even if the script still
contains an older fallback value.

## Examples

Dry run before submitting:

```bash
scripts/dev/sbatch_use_max_time.sh --dry-run SLURM/Auxme/auxme_gpu.sl
```

Override partition or QoS discovery when testing a variant:

```bash
scripts/dev/sbatch_use_max_time.sh --partition a30 --qos a30-gpu SLURM/Auxme/auxme_gpu.sl
```

Force a shorter manual wall time when needed:

```bash
scripts/dev/sbatch_use_max_time.sh --time 08:00:00 SLURM/Auxme/auxme_gpu.sl
```

Issue #749 imitation pipeline with a shared artifact root:

```bash
ISSUE749_RESULTS_DIR=output/slurm/issue749-pipeline-$(date +%Y%m%dT%H%M%S) \
ISSUE749_SOURCE_POLICY_ID=ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200 \
ISSUE749_TRAIN_SCRIPT=scripts/training/collect_expert_trajectories.py \
ISSUE749_TRAIN_ARGS="--dataset-id issue_749_v3_expert_traj_200 --policy-id ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200 --training-config configs/training/ppo/expert_ppo_issue_576_br06_v10_predictive_foresight_success_priority_policy_analysis_select.yaml --episodes 200 --seeds 123 231 777 992 1337" \
scripts/dev/sbatch_use_max_time.sh SLURM/Auxme/issue_749_imitation_pipeline.sl
```

Re-use the same `ISSUE749_RESULTS_DIR` for the BC and PPO fine-tuning follow-up jobs so
`benchmarks/expert_trajectories/` and `benchmarks/expert_policies/` stay visible across the
dependent submissions.

## Guidance

- Prefer the wrapper for long-running training jobs.
- Keep explicit short limits only for intentionally bounded jobs such as setup or quick
  interactive sessions.
- When adding a new batch script, include `#SBATCH --partition` and `#SBATCH --qos` so
  the wrapper can resolve the correct limit without extra flags.
- If Slurm tools are unavailable in the current shell, fall back to a manual `sbatch`
  command with an explicit `--time`.