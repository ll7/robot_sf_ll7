# Issue #856 - PPO all-scenarios full-budget campaign

Date: 2026-04-29

This note tracks the issue-856 follow-up that reruns the issue-791 leader recipe on the broader
training manifest at the full 10M budget.

## Goal

Measure whether the broad-training control closes the alignment-vs-diversity ambiguity left by the
eval-aligned issue-791 leader.

- Training manifest under test:
  `configs/scenarios/sets/ppo_all_available_training_v1.yaml`
- Leader reference:
  job 11724, success `0.929` on the eval-aligned in-distribution surface
- Camera-ready PPO reference:
  job 12122, `success_mean=0.2553`, `collisions_mean=0.0851`, `snqi_mean=-0.2906`,
  `time_to_goal_norm_mean=0.9274`

## Repo Changes For This Campaign

- Reused the existing seed-123 config:
  `configs/training/ppo/ablations/expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity.yaml`
- Added explicit seed-replica configs so the gated follow-up submissions stay reproducible:
  - `configs/training/ppo/ablations/expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity_seed231.yaml`
  - `configs/training/ppo/ablations/expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity_seed1337.yaml`
- Added the horizon-500 continuation surfaces for the best-checkpoint warm start:
  - `configs/scenarios/sets/ppo_all_available_training_v1_horizon500.yaml`
  - `configs/training/ppo/ablations/expert_ppo_issue_791_best_ckpt_all_scenarios_horizon500_20m_env22.yaml`

The original three all-scenarios configs keep the same PPO hyperparameters, `grid_socnav`
extractor, reward curriculum, predictive foresight settings, `num_envs: 22`, and
`randomize_seeds: true`. The horizon-500 continuation inherits the same env22 recipe and swaps the
scenario manifest plus resume/target-timestep settings.

## Submission Evidence

Seed-123 was submitted on 2026-04-29 through the maintained Auxme helper:

```bash
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity.yaml \
  --wandb-policy require \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

Observed helper output before submit:

- `a30`: `free_gpu=4`, `pending=0`, `slots_left=1`
- `l40s`: `free_gpu=0`, `pending=2`, `slots_left=1`
- Selected routing: `partition=a30`, `qos=a30-gpu`

Accepted job:

- Slurm job ID: `12172`
- Wrapper: `SLURM/Auxme/issue_791_reward_curriculum.sl`
- Effective wall time: `1-12:00:00`
- WandB policy: `require`
- Expected synced artifact root:
  `output/slurm/issue791-reward-curriculum-job-12172/`
- Expected stdout log:
  `output/slurm/12172-issue791-reward-curriculum.out`

Follow-up pro6000 rerun:

- Slurm job ID: `12223`
- Finished 2026-05-01 after 6h09m on `pro6000`.
- Best in-distribution eval at step `9,961,472 / 10,000,000`:
  `success_rate=0.900`, `collision_rate=0.100`, `snqi=0.226`.
- WandB: `ll7/robot_sf/ateif3c8`.
- Best checkpoint:
  `output/slurm/issue791-reward-curriculum-job-12223/benchmarks/expert_policies/checkpoints/ppo_expert_issue_791_all_scenarios_10m_env22_large_capacity/ppo_expert_issue_791_all_scenarios_10m_env22_large_capacity_best.zip`.

## Replica Gate

Do not queue the seed-231 and seed-1337 configs until job 12172 completes and the best checkpoint
is evaluated.

Gate for proceeding with replicas:

- If the broad-training seed-123 result lands more than `0.05` below the eval-aligned leader on
  the in-distribution surface, stop at one seed and report the result as a negative control.
- Otherwise submit the two replica configs above and compare their camera-ready benchmark band
  against the 11724/12122 references.

Gate status after job `12223`: **stop at one seed for now**. The broad-training rerun landed
`0.029` below the eval-aligned leader on success but `0.029` worse on collision rate. This is close
enough to justify one camera-ready comparison row, but not strong enough to spend two more full
10M training runs before benchmark evidence exists.

Queue-fill update on 2026-05-01: after the camera-ready benchmark completed successfully and l40s
was idle, the two staged 10M replica configs were dry-run validated and submitted to l40s for
long-running seed-band evidence. Two first submissions (`12255`, `12256`) accidentally inherited
the wrapper's `a30` partition and were canceled before starting. Corrected l40s jobs:

| Job | Config | Partition | Initial state |
|----:|--------|-----------|---------------|
| 12257 | `configs/training/ppo/ablations/expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity_seed231.yaml` | `l40s` | RUNNING on `auxme-imech091` |
| 12258 | `configs/training/ppo/ablations/expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity_seed1337.yaml` | `l40s` | RUNNING on `auxme-imech093` |

Both use the l40s maximum wall time of `3-00:00:00` and write logs to
`output/slurm/i856_allscen_10m_s{231,1337}_<jobid>.out`.

## Horizon-500 Best-Checkpoint Slurm Handoff

Date: 2026-05-05

Goal: continue the current issue-791 best checkpoint on the broad all-available scenario surface
with a uniform `max_episode_steps=500` horizon. This is a cluster-sized training run, not a local
desktop run.

Config:
`configs/training/ppo/ablations/expert_ppo_issue_791_best_ckpt_all_scenarios_horizon500_20m_env22.yaml`

This config:

- includes `configs/scenarios/sets/ppo_all_available_training_v1_horizon500.yaml`,
- resumes from model id `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`,
- starts from cached SB3 step `10272768`,
- targets final `total_timesteps: 20275200`, so the intended continuation is about 10M additional
  aggregate VecEnv transitions,
- inherits `num_envs: 22`, `worker_mode: subproc`, `grid_socnav`, large capacity, reward
  curriculum, and CUDA predictive foresight from
  `expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity.yaml`,
- writes W&B run name `ppo_issue791_best_ckpt_all_scenarios_horizon500_20m_env22`.

Local failed attempt:

- W&B run: `ll7/robot_sf/1btmjxks`
- Local log: `output/training_logs/issue791_horizon500/train_20260505T174326.log`
- Started on local host `auxme-imech036` at 2026-05-05 17:44:59 CEST.
- Crashed before the first scheduled evaluation/checkpoint.
- Root cause from kernel log:
  `Out of memory: Killed process 97076 (python3) ... anon-rss:10322924kB`.
- The later `BrokenPipeError` traces from `ForkServerProcess-*` workers are secondary fallout from
  the parent process being OOM-killed.

Reason for Slurm rerun:

The local machine has about 31 GiB RAM and the 22-env setup is too large for this
grid-socnav + predictive-foresight recipe. At the OOM event, the parent process was around 10 GiB
RSS and many subprocess environments were each several hundred MiB RSS, so the run exhausted host
memory almost immediately. The maintained Auxme wrapper requests `--mem=96G`, `--cpus-per-task=24`,
and one GPU, which is the intended resource envelope for this env22 configuration.

Validation before handoff:

```bash
uv run python - <<'PY'
from pathlib import Path
import yaml

for path in [
    Path("configs/scenarios/sets/ppo_all_available_training_v1_horizon500.yaml"),
    Path("configs/training/ppo/ablations/expert_ppo_issue_791_best_ckpt_all_scenarios_horizon500_20m_env22.yaml"),
]:
    yaml.safe_load(path.read_text(encoding="utf-8"))
    print(f"{path}: ok")
PY

tmpdir=$(mktemp -d /tmp/robot-sf-horizon500-dryrun.XXXXXX)
ROBOT_SF_ARTIFACT_ROOT="$tmpdir" LOGURU_LEVEL=WARNING \
  uv run python scripts/training/train_ppo.py \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_best_ckpt_all_scenarios_horizon500_20m_env22.yaml \
  --dry-run --log-level WARNING
```

Observed on 2026-05-05: both YAML files parsed, required Slurm/helper paths existed, and the
training dry-run exited `0`. The scenario-loader warnings about entries with both `map_id` and
`map_file` are inherited from the included broad-training scenario set and use `map_id`.

Recommended submission command:

```bash
git fetch origin
git checkout main
git pull --ff-only

scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_best_ckpt_all_scenarios_horizon500_20m_env22.yaml \
  --wandb-policy require \
  --job-name robot-sf-791-horizon500-bestckpt \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

If the helper chooses a congested partition, explicitly route to an open GPU partition:

```bash
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_best_ckpt_all_scenarios_horizon500_20m_env22.yaml \
  --partition l40s --qos l40s-gpu \
  --wandb-policy require \
  --job-name robot-sf-791-horizon500-bestckpt \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

Expected artifacts after a Slurm job `<jobid>`:

- stdout log: `output/slurm/<jobid>-issue791-reward-curriculum.out`
- synced artifact root: `output/slurm/issue791-reward-curriculum-job-<jobid>/`
- best-checkpoint directory:
  `output/slurm/issue791-reward-curriculum-job-<jobid>/benchmarks/expert_policies/checkpoints/ppo_expert_issue_791_best_ckpt_all_scenarios_horizon500_20m_env22/`

Early health checks after the job starts:

```bash
tail -f output/slurm/<jobid>-issue791-reward-curriculum.out
sstat -j <jobid>.batch --format=JobID,MaxRSS,AveRSS,MaxVMSize,AveCPU
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocCPUS,AllocTRES%40
```

Continue only if the log reaches the first scheduled evaluation at step `10485760` and produces a
checkpoint. If it OOMs again on `a30` or `l40s`, rerun on a larger-memory partition or make an
explicit reduced-env follow-up config instead of silently changing this env22 evidence surface.

## Camera-ready Benchmark Result

Job `12226` completed successfully on `pro6000`:

- campaign:
  `output/benchmarks/issue_856/paper_experiment_matrix_v1_issue_856_all_scenarios_compare_issue856-all-scenarios-12223-camera-ready_20260501_074754`
- runs: `7 / 7` successful
- episodes: `987`
- `benchmark_success=true`
- warnings: none
- runtime: about `1555s`

## Comparison vs Job 12122 (Eval-Aligned Leader)

Read directly from
`output/benchmarks/issue_856/.../reports/campaign_table_experimental.md` (broad-training PPO row,
141 episodes) against the note's recorded reference for job 12122 on the same camera-ready matrix:

| Metric | Broad-training (12223) | Eval-aligned reference (12122) | Δ (broad − eval) |
|---|---:|---:|---:|
| `success_mean` | 0.2199 | 0.2553 | −0.0354 |
| `collisions_mean` | 0.0922 | 0.0851 | +0.0071 |
| `snqi_mean` | −0.3305 | −0.2906 | −0.0399 |

Verdict: **broad-training underperforms the eval-aligned leader on every metric** of the
camera-ready matrix at fixed 10M budget. The arm is therefore not publication grade as a
replacement PPO row; it is a single-seed negative control that strengthens the
alignment-dominates claim already recorded in
[memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md](../../memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md).

Caveat: this is a single-seed comparison. The seed-band from replicas 12257 / 12258 is required
before treating the gap as statistically meaningful for the manuscript. The memory note records
the same caveat.

## Replica Seed-Band Write-Back

When jobs `12257` (seed 231) and `12258` (seed 1337) finish, write the outcome in two places:

1. Append a "Replica Seed-Band" subsection here with the per-seed best-checkpoint metrics, the
   per-seed PPO row of the camera-ready matrix, and the seed-band mean ± std vs the 12122
   reference.
2. Update
   [memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md](../../memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md)
   so the alignment-vs-diversity entry references the seed band rather than the single-seed
   point estimate.

If the seed band still underperforms the 12122 reference by more than `0.02` on `success_mean`,
do not promote the broad-training arm. If it falls within the 12122 seed band, promote the 12223
checkpoint to a durable store before opening any benchmark-facing PR (see "Durable Artifact
Decision" below).

## Durable Artifact Decision

The 12223 best checkpoint currently exists only at the worktree-local path
`output/slurm/issue791-reward-curriculum-job-12223/.../*_best.zip`, with provenance through
WandB run `ll7/robot_sf/ateif3c8`. Per the durable-artifact rule in
[AGENTS.md](../../AGENTS.md):

- The checkpoint stays as a worktree-local control while broad-training remains a single-seed
  negative result. The WandB run is the durable provenance handle.
- The checkpoint is promoted to a durable store (W&B model artifact, Zenodo, or model registry
  entry) only if the seed-band lands inside the 12122 reference band and the broad-training row
  becomes a publication candidate. The promotion step writes a `model/registry.yaml` entry whose
  artifact reference is the durable URL, not the local path.
- The 532K publication bundle at
  `output/benchmarks/issue_856/publication/issue856_all_scenarios_12223_12226_publication_bundle.tar.gz`
  is treated the same way: keep local until promotion, then upload as part of the same release
  step. No downstream workflow may depend on the local bundle path.

Queued three lightweight post-benchmark jobs on `pro6000` for provenance and publication handling:

| Job | Purpose | Output |
|----:|---------|--------|
| 12235 | campaign consistency/diagnostic analysis | `output/analysis/issue856/campaign_analysis_12226.{json,md}` |
| 12236 | benchmark artifact size report | `output/analysis/issue856/artifact_size_report_12226.json` |
| 12237 | no-video publication bundle export | `output/benchmarks/issue_856/publication/issue856_all_scenarios_12223_12226_publication_bundle*` |

All three completed on 2026-05-01. The publication bundle is local and intentionally untracked:
`output/benchmarks/issue_856/publication/issue856_all_scenarios_12223_12226_publication_bundle.tar.gz`
(`532K`, no videos, 48 files). Do not rely on this worktree-local `output/` path as a durable
source until it is uploaded to a release, Zenodo, W&B artifact, or another persistent store.

Artifact preservation decision:

- Keep the committed configs and this context note as the durable review surface.
- Leave generated `output/` files ignored in the worktree.
- Preserve the 12223 PPO checkpoint provenance through WandB run `ll7/robot_sf/ateif3c8` and the
  local adapter config `configs/baselines/ppo_issue_856_all_scenarios_12223.yaml`; the checkpoint
  itself remains at a local ignored path until explicitly promoted.
- Treat `output/benchmarks/expert_policies/ppo_expert_issue_791_all_scenarios_10m_env22_large_capacity_seed{231,1337}.zip`
  as disposable dry-run placeholders (`31B` each), not trained replica artifacts.

## Validation Path After Job 12172 Lands

1. Run the best-checkpoint in-distribution evaluation with
   `configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`.
2. Add a benchmark adapter config under `configs/baselines/` for the trained artifact.
   Done: `configs/baselines/ppo_issue_856_all_scenarios_12223.yaml`.
3. Rerun the dedicated broad-training matrix
   `configs/benchmarks/paper_experiment_matrix_v1_issue_856_all_scenarios_compare.yaml`.
   Done as job `12226`; benchmark completed successfully.
4. Compare the PPO row against job 12122 and decide whether the broad-training arm is publication
   grade, parity-only, or clearly worse.
5. Update
   `memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md`
   once the outcome is known.

## Metadata Limitation

Issue #856 is the tracked GitHub issue for this work and the current branch is
`856-ppo-all-scenarios`. A Project #5 status update was attempted from this environment, but the
installed `gh` token lacks the `read:project` scope needed to read or mutate `projectV2` fields.
The issue state can still be tracked through the branch, this note, and the queued Slurm job.

## Related Surfaces

- GitHub issue: `https://github.com/ll7/robot_sf_ll7/issues/856`
- `docs/context/issue_791_wave6_results_and_benchmark_orca_block.md`
- `docs/context/issue_791_best_policy_verdict_2026_04_21.md`
- `memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md`
