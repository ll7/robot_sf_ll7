---
name: auxme-issue791-submit
description: Submit issue-791-specific Auxme training jobs with explicit config provenance and wrapper-safety
  checks.
category: slurm
kind: atomic
phase: implementation
requires_write: true
requires_slurm: true
requires_benchmark_artifacts: false
delegates_to:
- slurm-campaign-submit
output_schema: campaign_submission.v1
---

# Auxme Issue 791 Submit

## When to use

Use this skill only for issue-791-specific Auxme training submissions that rely on `scripts/dev/sbatch_auxme_issue791.sh` or `ISSUE791_TRAIN_CONFIG`.

## Workflow

1. Read `SLURM/AGENTS.md`, `SLURM/Auxme/README.md`, and `docs/dev/slurm_submission.md`.
2. Confirm the explicit config path and reject wrapper-default config use.
3. Check live partition pressure with `scripts/dev/auxme_partition_status.sh --recommend`.
4. Submit with `scripts/dev/sbatch_auxme_issue791.sh --config <path> --job-name <name> SLURM/Auxme/<script>.sl`.
5. Capture job id, stdout/stderr paths, commit SHA, config path, and expected artifact root.
6. Classify transient infrastructure failures separately from training or config failures.

## Guardrails

- Do not use this for non-issue-791 campaigns; route generic jobs to `slurm-campaign-submit`.
- Never submit without an explicit config path.
- Do not count submission as benchmark evidence; it is only provenance for a running campaign.

## Output

Use `campaign_submission.v1`.
