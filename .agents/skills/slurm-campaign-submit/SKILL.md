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
3. Run the SLURM suitability gate before any `sbatch`:
   - estimate rows, seeds, scenarios, episodes, horizon, workers, expected walltime, GPU need, and
     expected artifacts from the config or preflight output;
   - classify the planned run as `local-smoke`, `compute-node-smoke`, `slurm-campaign`, or
     `blocked-needs-scope`;
   - default to local execution for jobs expected to finish in under 1 hour unless they require a
     compute-node-only dependency, GPU, queue/runtime parity proof, or explicit maintainer approval;
   - when a short run is submitted to SLURM anyway, label it as a smoke/probe in the job label,
     handoff note, issue comment, and PR text, not as completed campaign evidence.
4. Run any campaign-specific launch-packet validator before `sbatch` when one exists under `scripts/validation/`.
5. Submit with explicit config and job name; capture job ID plus stdout/stderr paths.
6. Record output root and expected artifacts: manifest, checkpoint, report, metrics, videos, or release bundle.
7. Classify status as `submitted`, `blocked`, or `failed_preflight`; classify failures as config, cluster capacity, wrapper, dependency, or unknown.
8. Hand artifact classification to `artifact-provenance` before downstream reports depend on local `output/`.

## Guardrails

- Do not infer live cluster capacity from stale logs.
- Do not submit from an unintended branch or dirty tree without calling that out.
- Do not treat a queued job as completed evidence.
- Do not let an issue's `slurm` label override the suitability gate. A `slurm` label means the
  issue may need cluster execution, not that every bounded smoke must be submitted.
- Do not call a short compatibility run a campaign-sized result. If the estimated or observed
  runtime is under 1 hour, preserve it as smoke/probe evidence unless the issue explicitly asks for
  a short compute-node proof.
- Do not upload artifacts externally unless the user explicitly requests it.

## Output

Use `campaign_submission.v1`. Include:

- `slurm_suitability`: one of `local-smoke`, `compute-node-smoke`, `slurm-campaign`, or
  `blocked-needs-scope`;
- `estimated_runtime_minutes`: the estimated runtime of the campaign in minutes;
- `runtime_basis`: the rows/seeds/scenarios/horizon/workers calculation or preflight source;
- `local_default_overridden`: `true` only when a sub-1-hour run still has a compute-node-only reason;
- `evidence_claim`: `smoke`, `compatibility`, `campaign`, or `blocked`.
