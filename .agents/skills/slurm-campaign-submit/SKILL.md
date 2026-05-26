---
name: slurm-campaign-submit
description: Submit generic SLURM campaigns with preflight, config provenance, job metadata, artifact
  expectations, and failure classification.
category: slurm
kind: atomic
phase: implementation
requires_write: true
requires_slurm: true
requires_benchmark_artifacts: false
delegates_to:
- artifact-provenance
output_schema: campaign_submission.v1
---

# SLURM Campaign Submit

## When to use

Use this skill for generic SLURM campaign submission: learned-risk, shielded PPO, ORCA-residual BC, predictive comparison, oracle imitation, adversarial search, and benchmark campaigns.

## Workflow

1. Read local cluster guidance (`local.machine.md` if present, then `SLURM/AGENTS.md` and `docs/dev/slurm_submission.md`).
2. Confirm config path, command surface, current commit SHA, and dirty-tree risk.
3. Run any campaign-specific launch-packet validator before `sbatch` when one exists under `scripts/validation/`.
4. Submit with explicit config and job name; capture job ID plus stdout/stderr paths.
5. Record output root and expected artifacts: manifest, checkpoint, report, metrics, videos, or release bundle.
6. Classify status as `submitted`, `blocked`, or `failed_preflight`; classify failures as config, cluster capacity, wrapper, dependency, or unknown.
7. Hand artifact classification to `artifact-provenance` before downstream reports depend on local `output/`.

## Guardrails

- Do not infer live cluster capacity from stale logs.
- Do not submit from an unintended branch or dirty tree without calling that out.
- Do not treat a queued job as completed evidence.
- Do not upload artifacts externally unless the user explicitly requests it.

## Output

Use `campaign_submission.v1`.
