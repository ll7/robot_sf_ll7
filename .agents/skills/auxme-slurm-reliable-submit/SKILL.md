---
name: auxme-slurm-reliable-submit
description: "Submit issue-791 style Auxme SLURM jobs with explicit config, live partition pressure checks, and max-time-safe wrapper routing."
---

# Auxme SLURM Reliable Submit

Use this skill for Auxme cluster submissions where reliability matters more than raw speed.

## Read First

- `SLURM/AGENTS.md`
- `SLURM/Auxme/README.md`
- `docs/dev/slurm_submission.md`

## Goals

- avoid accidental stage1 fallback caused by missing config overrides,
- choose partition/QoS using live availability and per-user slot headroom,
- keep wall-time aligned with current partition policy.

## Workflow

1. Confirm the target config and intent
   - Validate the YAML path exists under `configs/training/...`.
   - Ensure horizon intent matches request (`32k`, `128k`, `1m`, `10m`, etc.).

2. Snapshot partition pressure
   - Run `scripts/dev/auxme_partition_status.sh`.
   - Use free GPUs, pending depth, and per-user running slots as the primary signals.

3. Submit through the reliable helper
   - Run `scripts/dev/sbatch_auxme_issue791.sh` with explicit `--config`.
   - Let it auto-select partition/QoS unless there is a deliberate override.

4. Verify startup provenance
   - Check job stdout startup summary for the exact config path and policy ID.
   - Fail fast if wrapper-default config appears.

5. Handle transient infra failures
   - If stdout shows `Unable to confirm allocation ... Zero Bytes were transmitted or received`,
     classify as transient and resubmit once with identical config.

## Commands

Status and recommendation:

```bash
scripts/dev/auxme_partition_status.sh
scripts/dev/auxme_partition_status.sh --recommend
```

Reliable submit:

```bash
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22.yaml \
  --job-name robot-sf-issue791-reward-curriculum \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

## Guardrails

- Do not submit issue-791 wrappers without explicit `ISSUE791_TRAIN_CONFIG`.
- Do not assume partition health from stale snapshots; always refresh right before submit.
- Do not classify allocation-handshake failures as model regressions.
