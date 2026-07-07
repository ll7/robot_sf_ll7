# Agent Runbook

## Standard Loop

1. Pick a candidate from `candidate_registry.yaml`.
2. Run `smoke` locally.
3. If smoke passes, run `nominal_sanity` locally.
4. If nominal sanity passes, run `stress_slice` locally.
5. Capture the result in a markdown report under `reports/`.
6. If the next stage is `full_matrix` or `robustness_extension`, create or update
   the matching handoff entry under `SLURM/` instead of running it on the laptop.

## Required Artifacts Per Run

- candidate name and config path,
- stage name,
- exact command,
- git commit,
- aggregate metrics,
- classic versus Francis split,
- failure taxonomy counts,
- short interpretation and next action.

## Local Boundary

Do not run `full_matrix` or `robustness_extension` locally unless the caller
explicitly overrides the stage guard.
