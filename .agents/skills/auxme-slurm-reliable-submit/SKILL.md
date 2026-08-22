---
name: auxme-slurm-reliable-submit
description: Submit issue-791 style Auxme SLURM jobs with explicit config, live partition pressure checks,
  and max-time-safe wrapper routing.
category: slurm
kind: atomic
phase: implementation
requires_write: true
requires_slurm: true
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
aliases:
- auxme-issue791-submit
- auxme-issue791-reliable-submit
---

# Auxme SLURM Reliable Submit

## When to use

Use this skill for issue-791-style Auxme submissions when the private operations overlay,
explicit config, live partition pressure, and wrapper-safe routing are required. Use the
`issue-791` profile for the legacy Issue 791 contract; route generic campaigns to
`slurm-campaign-submit`.

## Modes

- `issue-791` profile: use the legacy `auxme-issue791-submit` alias or select this named profile
  for Issue 791-style Auxme training jobs.
- Generic campaigns: route to `slurm-campaign-submit`; this skill does not broaden its wrapper
  contract to unrelated campaigns.

The `auxme-issue791-submit` and `auxme-issue791-reliable-submit` names are compatibility aliases
for this canonical skill. The canonical skill never delegates back to an alias.

## Purpose

Submit Auxme jobs for issue-791-style training reliably and reproducibly.
Use this when reliability, provenance, and correct config routing matter more than raw queue speed.

### Issue-791 profile

Use this profile only for Issue 791-specific Auxme submissions that rely on
`scripts/dev/sbatch_auxme_issue791.sh` or `ISSUE791_TRAIN_CONFIG`.

The profile preserves the legacy `campaign_submission.v1` output contract. It is a submission
provenance result, not benchmark evidence.

#### Workflow

1. Read cluster-specific preflight and confirm the private operations overlay is configured:
   - `SLURM/AGENTS.md`
   - `SLURM/Auxme/README.md`
   - `docs/dev/slurm_submission.md`
2. Confirm target intent:
   - Validate the `--config` path exists under `configs/training/...`.
   - Verify requested training horizon (`32k`, `128k`, `1m`, `10m`) matches the user request.
3. Check live capacity before submit:
   - `scripts/dev/auxme_partition_status.sh`
   - `scripts/dev/auxme_partition_status.sh --recommend`
   - Use free GPUs, pending depth, and per-user running slots only from the live output.
4. Submit with explicit config:
   - `scripts/dev/sbatch_auxme_issue791.sh --config <path> --job-name <name> SLURM/Auxme/<script>.sl`
5. Verify startup:
   - Inspect stdout for exact config path and policy ID.
   - Fail fast if wrapper-default config is used.
6. Transient failure handling:
   - If allocation handshake shows `Zero Bytes were transmitted or received`, retry once with identical arguments.

#### Guardrails

- Never submit an issue-791 wrapper without explicit `ISSUE791_TRAIN_CONFIG`/`--config`.
- Do not use stale partition status for a submission decision.
- Do not interpret infrastructure handshake failures as training quality regressions.
- Do not use this profile for non-issue-791 campaigns; route generic jobs to `slurm-campaign-submit`.

#### Output

- Chosen config path, partition/QoS decision, submit command.
- Startup provenance (stdout markers) and whether a retry was triggered.
- Final outcome (`submitted` / `blocked` / `retry suggested`) with exact reason.
- Schema: `campaign_submission.v1`.

## Canonical output

For callers that do not select the Issue-791 profile, return `skill_run_summary.v1` with the
selected mode, proof status, and any routing blocker. No profile may silently change the wrapper,
config, or campaign evidence contract.
