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
2. Before choosing a job, run the discovery gate:
   - refresh live user jobs with `squeue --me` and recent outcomes with `sacct`;
   - query open GitHub issues with `resource:slurm` or `slurm` labels, including comments when an
     issue may be runnable;
   - compare labels against the canonical SLURM ledger and recent context notes rather than trusting
     labels alone;
   - classify each candidate as `ready_to_submit`, `blocked_dependency`, `analysis_only`,
     `already_running`, or `completed_needs_analysis`;
   - do not submit when the only unblocked item already has completed results that need analysis.
3. Confirm config path, command surface, current commit SHA, and dirty-tree risk.
4. Run the SLURM suitability gate before any `sbatch`:
   - estimate rows, seeds, scenarios, episodes, horizon, workers, expected walltime, GPU need, and
     expected artifacts from the config or preflight output;
   - classify the planned run as `local-smoke`, `compute-node-smoke`, `slurm-campaign`, or
     `blocked-needs-scope`;
   - default to local execution for jobs expected to finish in under 1 hour unless they require a
     compute-node-only dependency, GPU, queue/runtime parity proof, or explicit maintainer approval;
   - when a short run is submitted to SLURM anyway, label it as a smoke/probe in the job label,
     handoff note, issue comment, and PR text, not as completed campaign evidence.
5. For long-running training submissions, confirm the experiment card or launch packet predeclares
   early-stop criteria: metric, threshold, check cadence, minimum runtime or timesteps, cancel
   condition, and diagnostic-preservation action. If these are missing, classify the candidate as
   `blocked-needs-scope` until the card is updated.
6. Run any campaign-specific launch-packet validator before `sbatch` when one exists under `scripts/validation/`.
7. For queue-backed training jobs, run
   `uv run python scripts/dev/submit_training_jobs.py --dry-run` and submit through
   `uv run python scripts/dev/submit_training_jobs.py --submit` only when the generated manifest
   shows the intended config, launcher, job name, output root, and duplicate-check evidence.
8. Submit non-queue campaigns with explicit config and job name; capture job ID plus stdout/stderr paths.
9. Record output root and expected artifacts: manifest, checkpoint, report, metrics, videos, or release bundle.
10. Classify pre-health status as `route_accepted`, `blocked`, or `failed_preflight`; classify
    successful route acceptance with missing issue/PR or private-ledger traceability as
    `partial_traceable`; classify failures as config, cluster capacity, wrapper, dependency, or unknown.
11. Immediately run a submission health check (`squeue`, `sacct`, initial stderr availability) and apply the
    shared checklist in `docs/dev/slurm_submission.md`.
12. Hand artifact classification to `artifact-provenance` before downstream reports depend on local `output/`.
13. A clean `submitted` state requires both immediate health-check success and the checklist traceability update
    (issue/PR comment plus private-ledger or handoff reference). If route acceptance succeeds but
    either traceability record is missing, keep `partial_traceable`. `sbatch`/`sacct` acceptance alone is only
    route evidence and does not imply benchmark or report proof.
14. After completion or predeclared cancellation, route results before rerunning:
    - cancelled runs may be diagnostic evidence only when the early-stop criteria were declared and
      the configured preservation action captured logs, manifest, config, commit, and artifact
      status;
    - unplanned cancellations stay `failed_preflight`, `blocked`, or `inconclusive` until triaged;
    - `completed_needs_analysis` goes to the
    relevant analysis skill or context note, while `failed_preflight` or early wrapper failures must
    be fixed in a worktree before resubmission.

## Guardrails

- Do not infer live cluster capacity from stale logs.
- Do not bypass `scripts/dev/submit_training_jobs.py` for queue-backed training jobs unless the
  helper cannot represent the launcher; record the exception and the equivalent duplicate checks.
- Do not submit from an unintended branch or dirty tree without calling that out.
- Do not treat a queued job as completed evidence.
- Do not cancel a low-signal training job as diagnostic evidence unless the early-stop rule was
  predeclared and the diagnostic-preservation action has been completed.
- Do not treat `sbatch` output or acceptance as final submission proof.
- Do not treat a `resource:slurm` issue as runnable when its latest comments require durable
  artifact pointers, exact commits, maintainer confirmation, or a concrete launcher first.
- Do not let an issue's `slurm` label override the suitability gate. A `slurm` label means the
  issue may need cluster execution, not that every bounded smoke must be submitted.
- Do not call a short compatibility run a campaign-sized result. If the estimated or observed
  runtime is under 1 hour, preserve it as smoke/probe evidence unless the issue explicitly asks for
  a short compute-node proof.
- Do not upload artifacts externally unless the user explicitly requests it.

## Output

Use `campaign_submission.v1`. Include:

- `discovery_state`: one of `ready_to_submit`, `blocked_dependency`, `analysis_only`,
  `already_running`, or `completed_needs_analysis`;
- `dependency_basis`: issue comment, ledger row, config validator, artifact pointer, or maintainer
  decision that controls the route;
- `slurm_suitability`: one of `local-smoke`, `compute-node-smoke`, `slurm-campaign`, or
  `blocked-needs-scope`;
- `estimated_runtime_minutes`: the estimated runtime of the campaign in minutes;
- `runtime_basis`: the rows/seeds/scenarios/horizon/workers calculation or preflight source;
- `local_default_overridden`: `true` only when a sub-1-hour run still has a compute-node-only reason;
- `evidence_claim`: `smoke`, `compatibility`, `campaign`, or `blocked`.
- `submission_record`: issue/PR reference, partition, finalizer outcome, private ledger reference,
  final health-check status, and artifact status.
- `submission_state`: `submitted`, `partial_traceable`, or `blocked`.
