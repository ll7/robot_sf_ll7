# SLURM Multi-Worktree Branch Workflow

## Goal

When multiple Robot SF branches need to submit or monitor SLURM jobs from the same login node,
use separate Git worktrees instead of switching one working tree back and forth. This keeps each
submitted job tied to the intended branch, config files, scripts, and untracked local setup in the
submit directory.

This note was written from the `auxme-imech192` login-node context on 2026-04-16. The local policy
there is to use the node for lightweight orchestration only and submit real compute through SLURM.

## Recommended Pattern

Create one worktree per active branch:

```bash
cd /home/luttkule/git/robot_sf_ll7
git fetch origin codex/193-feature-extractor-evaluation
git worktree add -b codex/193-feature-extractor-evaluation \
  ../robot_sf_ll7_193_feature_extractor \
  origin/codex/193-feature-extractor-evaluation
cd ../robot_sf_ll7_193_feature_extractor
```

If the local branch already exists, omit `-b codex/193-feature-extractor-evaluation` and pass the
local branch name as the final argument instead.

Then submit branch-specific jobs from inside that worktree:

```bash
scripts/dev/sbatch_use_max_time.sh SLURM/feature_extractor_comparison/run_comparison.slurm
```

For issue-791 jobs, stay in the issue-791 worktree and use the explicit-config helper:

```bash
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22.yaml \
  --job-name robot-sf-issue791-reward-curriculum \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

## Why Worktrees Are Preferred

- SLURM jobs inherit the submit directory through `SLURM_SUBMIT_DIR`; many repository wrappers use
  that path, or resolve the Git root from it, before reading configs and scripts.
- A branch switch in a single checkout can make local inspection confusing while older jobs still
  reflect the code they copied or resolved from the original submit path.
- Separate worktrees let VS Code, terminals, logs, and pending job reasoning stay branch-scoped.
- Worktrees share Git object storage, so they are cheaper and less error-prone than full duplicate
  clones.

Important boundary: a worktree is not a submission snapshot. If a job is pending, it will normally
read files from that worktree when the job actually starts. Commit or otherwise freeze the configs
and scripts that matter, and avoid incompatible edits in that worktree until the queued job has
started or copied its run files.

## Local Machine Context

`local.machine.md` is intentionally gitignored, so new worktrees will not receive it from Git.
When the same machine policy should apply to every worktree on the same host, symlink it:

```bash
ln -s ../robot_sf_ll7/local.machine.md ../robot_sf_ll7_193_feature_extractor/local.machine.md
```

Use a relative symlink when both worktrees live under the same parent directory. If the target
worktree lives elsewhere, use an absolute path instead.

## Virtual Environment Boundary

Keep dependency environments branch-local unless you are certain the branches have identical
dependency requirements. Most SLURM scripts expect `.venv` under the submit worktree, so a new
worktree usually needs its own setup:

```bash
uv sync --all-extras --frozen
```

Avoid symlinking `.venv` across branches by default. Sharing a virtualenv can hide dependency
drift and make a job appear to run on one branch while using dependencies last synced for another.
The UV package cache can still be shared globally, so separate `.venv` directories do not imply a
full redownload every time.

## Operational Notes

- Open a separate VS Code window per worktree when actively editing or submitting from both
  branches.
- Give jobs branch-specific names where possible so `squeue -u "$USER"` remains readable.
- Keep per-branch outputs in branch- or issue-named subdirectories under `output/`; do not write
  new artifacts to legacy `results/` paths.
- Check queue pressure from any worktree, but submit from the worktree whose code and configs the
  job should execute.
- If a job script still hardcodes `cd $SLURM_SUBMIT_DIR`, run `sbatch` or the wrapper from the
  repository root of the intended worktree.
- For long pending jobs, record the commit SHA and key config path in the job name, log, context
  note, or issue comment so later inspection can reconstruct what was meant to run.

## Related Surfaces

- [SLURM Submission Workflow](../dev/slurm_submission.md)
- [Local machine context template](../templates/local.machine.example.md)
- [Issue 791 Promotion Campaign](issue_791_promotion_campaign_128k_256k.md)
