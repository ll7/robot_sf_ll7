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

## Guidance

- Prefer the wrapper for long-running training jobs.
- Keep explicit short limits only for intentionally bounded jobs such as setup or quick
  interactive sessions.
- When adding a new batch script, include `#SBATCH --partition` and `#SBATCH --qos` so
  the wrapper can resolve the correct limit without extra flags.
- If Slurm tools are unavailable in the current shell, fall back to a manual `sbatch`
  command with an explicit `--time`.

## Multiple branches from one login node

When two active branches need to submit or monitor SLURM jobs from the same login node, prefer one
Git worktree per branch. Submit from the worktree whose branch, configs, and SLURM scripts should
be used by the job:

```bash
cd /home/luttkule/git/robot_sf_ll7
git fetch origin codex/193-feature-extractor-evaluation
git worktree add -b codex/193-feature-extractor-evaluation \
  ../robot_sf_ll7_193_feature_extractor \
  origin/codex/193-feature-extractor-evaluation
cd ../robot_sf_ll7_193_feature_extractor
scripts/dev/sbatch_use_max_time.sh SLURM/feature_extractor_comparison/run_comparison.slurm
```

This is safer than switching one checkout between branches while jobs are pending because SLURM
sets `SLURM_SUBMIT_DIR` to the directory where `sbatch` was called, and repository wrappers often
use that directory or resolve the Git root from it before reading configs.

This isolates branches, not file snapshots. Pending jobs normally read the worktree contents when
they start, so avoid incompatible edits to that worktree's configs or scripts while a queued job is
waiting.

`local.machine.md` is gitignored. If the same login-node policy should apply to every local
worktree, symlink it from the original checkout:

```bash
ln -s ../robot_sf_ll7/local.machine.md ../robot_sf_ll7_193_feature_extractor/local.machine.md
```

Keep `.venv` branch-local unless the branches are known to have identical dependencies; most SLURM
scripts expect `.venv` under the submit worktree. See the durable workflow note:
[SLURM Multi-Worktree Branch Workflow](../context/slurm_multi_worktree_branch_workflow.md).

## Auxme issue-791 reliability helper

For issue-791 wrappers on Auxme, use:

```bash
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22.yaml \
  --job-name robot-sf-issue791-reward-curriculum \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

This helper adds pre-submit partition availability checks and recommendation logic based on
current cluster pressure, then submits through `sbatch_use_max_time.sh`.

Raw status table only:

```bash
scripts/dev/auxme_partition_status.sh
```

Machine-readable recommendation only:

```bash
scripts/dev/auxme_partition_status.sh --recommend
```
