---
name: auxme-slurm-reliable-submit
description: "Submit issue-791 style Auxme SLURM jobs with explicit config, live partition pressure checks, and max-time-safe wrapper routing."
---

# Auxme SLURM Reliable Submit

## Purpose

Submit Auxme jobs for issue-791-style training reliably and reproducibly.
Use this when reliability, provenance, and correct config routing matter more than raw queue speed.

## Workflow

1. Read cluster-specific preflight:
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

## Guardrails

- Never submit an issue-791 wrapper without explicit `ISSUE791_TRAIN_CONFIG`/`--config`.
- Do not use stale partition status for a submission decision.
- Do not interpret infrastructure handshake failures as training quality regressions.

## Output

- Chosen config path, partition/QoS decision, submit command.
- Startup provenance (stdout markers) and whether a retry was triggered.
- Final outcome (`submitted` / `blocked` / `retry suggested`) with exact reason.
