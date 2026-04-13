# SLURM Agent Playbook

Use this file as the canonical SLURM execution playbook for reusable cluster workflows in this
repository. Pair it with [AGENTS.md](../AGENTS.md), [SLURM/readme.md](readme.md), and
[docs/dev/slurm_resource_audit.md](../docs/dev/slurm_resource_audit.md).

For Auxme-cluster-specific details (partitions, QoS profiles, per-user job limits, and
`num_envs` guidance), see [SLURM/Auxme/README.md](Auxme/README.md).

## Scope

Apply these rules whenever submitting or reviewing jobs under [SLURM/](.) or related training
campaigns that run on cluster resources.

## Submission Defaults

- Prefer repo-local wrappers and config-first commands.
- Use `ISSUE*_TRAIN_CONFIG`-style environment selection and keep configs under
  [configs/training/](../configs/training/).
- For long jobs, prefer `scripts/dev/sbatch_use_max_time.sh` over raw `sbatch` unless the task
  explicitly requires fixed wall time.
- Keep artifact output rooted in `output/slurm/` (or the configured mirrored destination) and
  confirm `ROBOT_SF_ARTIFACT_ROOT` is synced on exit.
- Set `#SBATCH --output=output/slurm/%j-<description>.out` — job ID first so logs sort
  chronologically and are gitignored via the root `output/` rule. Never write `.out` files
  to the repo root.

## Tracking Policy (WandB)

- Promotion/follow-up runs must keep `tracking.wandb.enabled: true`.
- Stage-gate debug runs may allow WandB off only when explicitly justified.
- For issue-791 style wrappers, use:
  - `ISSUE791_WANDB_POLICY=require` for promotion/follow-up,
  - `ISSUE791_WANDB_POLICY=allow-off` only for explicit debug gates.
- Never submit long runs without verifying run group, job type, and tags in config.

## Runtime Sanity Checks

Before submission:

1. Verify the selected YAML exists and is the intended horizon (`32k`, `128k`, `1m`, etc.).
2. Verify `num_envs` matches host strategy (see [Auxme/README.md](Auxme/README.md) for
   per-cluster guidance; choose intentionally).
3. Verify eval cadence (`evaluation.step_schedule`) is dense enough for decision quality but not so
   dense that wall-clock is dominated by evaluation.

After submission:

1. Check acceptance via `squeue` and record job IDs.
2. Check job accounting via `sacct` for `COMPLETED` vs infrastructure failure signatures.
3. Check stdout tail for immediate allocation errors (for example,
   `Unable to confirm allocation ... Zero Bytes were transmitted or received`).

## Failure Signatures and Responses

- Allocation handshake failure (`srun` unable to confirm allocation):
  - classify as infrastructure/transient,
  - resubmit once with same config,
  - do not treat as model failure.
- Missing artifacts in expected output path:
  - verify `ROBOT_SF_ARTIFACT_ROOT` and cleanup sync path,
  - do not conclude run had no results before checking `/tmp/<user>/<jobid>/results` style roots.
- NaN or evaluation instability:
  - preserve logs and minimal repro evidence,
  - block promotion claims until finite deterministic evaluation is restored.

## Campaign Progression Rules

- Do not claim performance improvement from short gate runs alone.
- Use staged progression and baseline anchoring:
  1. gate (`8k` to `32k`) for feasibility and crash checks,
  2. promotion (`128k` to `256k`) for early ranking,
  3. long horizon (`1m+`) for convergence-level decisions.
- Keep at least one unchanged baseline run in each campaign wave.

## Required Insight Capture (Between Sessions)

Every time a SLURM run yields new evidence, persist it before closing the task.

Required capture targets:

1. Update this file when the insight changes reusable SLURM practice.
2. Update issue/campaign context notes under [docs/context/](../docs/context/).
3. Update results ledger files (for example,
   [output/ai/autoresearch/issue-791/results.tsv](../output/ai/autoresearch/issue-791/results.tsv)).

Examples of insights that must be saved:

- best `num_envs`/CPU pairings by host,
- wall-clock behavior by training horizon,
- reliable eval cadence ranges,
- recurrent failure signatures and the proven mitigations,
- WandB policy pitfalls and enforcement updates.

A SLURM task is not complete until at least one persistent insight surface was updated when new
information was discovered.
