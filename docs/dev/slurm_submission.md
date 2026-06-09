# SLURM Submission Workflow

[← Back to Documentation Index](../README.md)

Use the shared wrapper below for new batch jobs so the requested wall time tracks the
current partition and QoS policy instead of relying on stale hardcoded `#SBATCH --time`
lines.

## Default workflow

```bash
scripts/dev/sbatch_use_max_time.sh <cluster-script.sl>
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
scripts/dev/sbatch_use_max_time.sh --dry-run <cluster-script.sl>
```

Override partition or QoS discovery when testing a variant on a configured cluster:

```bash
scripts/dev/sbatch_use_max_time.sh \
  --partition <partition> --qos <qos> \
  --sbatch-arg --partition=<partition> \
  --sbatch-arg --qos=<qos> \
  SLURM/templates/gpu_training.sl
```

Force a shorter manual wall time when needed:

```bash
scripts/dev/sbatch_use_max_time.sh --time 08:00:00 SLURM/templates/gpu_training.sl
```

## Guidance

- Prefer the wrapper for long-running training jobs.
- Keep explicit short limits only for intentionally bounded jobs such as setup or quick
  interactive sessions.
- Before submitting, estimate the intended runtime from the config or preflight output: rows,
  scenarios, seeds, episodes, horizon, workers, GPU need, and expected artifacts. A benchmark or
  training run expected to finish in under 1 hour should default to local execution, not SLURM.
- Submit a sub-1-hour run to SLURM only when the point is compute-node proof: GPU-only execution,
  cluster-only dependencies, queue/runtime parity, or a maintainer-approved smoke. Label that job
  and any issue/PR follow-up as `smoke` or `probe`, not as completed campaign evidence.
- Do not let a GitHub `slurm` label alone justify submission. The label means cluster execution may
  be required; the config still needs a suitability check.
- For long-running training jobs, include predeclared early-stop criteria in the experiment card or
  launch packet before submission. The criteria must name the metric, threshold, check cadence,
  minimum runtime or timesteps, exact cancel condition, and diagnostic-preservation action.
- A cancelled Slurm run can be valid diagnostic evidence only when the stop rule was predeclared and
  the branch, commit, config, logs, manifest, local output root, and durable artifact/preservation
  status are recorded. Without that preservation trail, classify the run as failed, blocked, or
  inconclusive rather than proof.
- When adding a new batch script, include `#SBATCH --partition` and `#SBATCH --qos` so
  the wrapper can resolve the correct limit without extra flags.
- If Slurm tools are unavailable in the current shell, fall back to a manual `sbatch`
  command with an explicit `--time`.

## Multiple branches from one login node

When two active branches need to submit or monitor SLURM jobs from the same login node, prefer one
Git worktree per branch. Submit from the worktree whose branch, configs, and SLURM scripts should
be used by the job:

```bash
cd ~/git/robot_sf_ll7
mkdir -p ../robot_sf_ll7.worktrees
git fetch origin codex/193-feature-extractor-evaluation
git worktree add -b codex/193-feature-extractor-evaluation \
  ../robot_sf_ll7.worktrees/codex-193-feature-extractor-evaluation \
  origin/codex/193-feature-extractor-evaluation
cd ../robot_sf_ll7.worktrees/codex-193-feature-extractor-evaluation
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
ln -s ../../robot_sf_ll7/local.machine.md local.machine.md
```

Keep `.venv` branch-local unless the branches are known to have identical dependencies; most SLURM
scripts expect `.venv` under the submit worktree. See the durable workflow note:
[SLURM Multi-Worktree Branch Workflow](../context/slurm_multi_worktree_branch_workflow.md).

## Private Cluster Overlays

Cluster-specific hostnames, QoS policies, node-packing heuristics, local scratch paths, and
machine-only runbooks should live outside this public repository. The public repo keeps the
portable experiment contract: checked-in configs, generic wrapper behavior, artifact policy,
validation helpers, and reviewable evidence manifests.

Configure the optional private operations overlay with either an environment variable:

```bash
export ROBOT_SF_PRIVATE_OPS=/path/to/robot_sf_ll7-private-ops
```

or a gitignored local machine context entry:

```markdown
- private_ops_repo: /path/to/robot_sf_ll7-private-ops
```

When neither is set, `scripts/dev/private_ops.sh` falls back to a sibling checkout named
`robot_sf_ll7-private-ops` next to the current worktree's parent directory.

For worktrees, prefer one sibling private overlay shared by all checkouts:

```text
~/git/robot_sf_ll7/
~/git/robot_sf_ll7.worktrees/<branch>/
~/git/robot_sf_ll7-private-ops/
```

## Auxme issue-791 private helper

For issue-791 wrappers on Auxme, use:

```bash
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22.yaml \
  --job-name robot-sf-issue791-reward-curriculum \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

This public helper delegates to the private operations overlay. The private implementation adds
pre-submit partition availability checks and recommendation logic based on current cluster pressure,
then submits through the public `sbatch_use_max_time.sh` in the active worktree.

Raw status table only:

```bash
scripts/dev/auxme_partition_status.sh
```

Machine-readable recommendation only:

```bash
scripts/dev/auxme_partition_status.sh --recommend
```

## Camera-ready benchmark campaigns

For camera-ready benchmark campaigns on a private cluster, prefer a generic launcher in the private
overlay rather than cloning an issue-specific public script:

```bash
CAMERA_READY_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml \
CAMERA_READY_BENCHMARK_LABEL=issue999-preflight \
CAMERA_READY_BENCHMARK_MODE=preflight \
scripts/dev/sbatch_use_max_time.sh --dry-run <private-camera-ready-benchmark.sl>
```

Submit the full run by removing `--dry-run` and setting the intended artifact root:

```bash
CAMERA_READY_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml \
CAMERA_READY_BENCHMARK_LABEL=issue999-camera-ready \
CAMERA_READY_BENCHMARK_OUTPUT_ROOT=output/benchmarks/issue_999 \
scripts/dev/sbatch_use_max_time.sh <private-camera-ready-benchmark.sl>
```

`CAMERA_READY_BENCHMARK_MODE=preflight` and `run` are both supported. The launcher requires an
explicit config and either `CAMERA_READY_BENCHMARK_LABEL` or `CAMERA_READY_BENCHMARK_CAMPAIGN_ID`
so queued jobs have a reviewable identity before they consume cluster time. Slurm logs stay under
`output/slurm/`; campaign outputs should stay under `output/benchmarks/...` unless a small
manifest, summary, or durable artifact pointer is intentionally promoted.
