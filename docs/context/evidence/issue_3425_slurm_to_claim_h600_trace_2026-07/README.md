# h600 SLURM-to-Claim Trace Checklist

This directory contains the public-safe Simple Linux Utility for Resource Management
(SLURM)-to-claim workflow trace checklist for h600 jobs `13268` and `13273`. Issue #4243
completes the finalizer and reconciliation links that the original issue #3425 trace left blocked.

## Claim Boundary

- Status: `pass`.
- Decision: `keep_diagnostic`.
- Scope: workflow/tooling trace only, using existing compact h600 evidence.
- Not claimed: planner ranking, benchmark conclusion, paper claim, dissertation claim, or full
  campaign readiness.

## Contents

- `finalizer_13268.json` and `finalizer_13273.json`: compact finalizer manifests with checksums
  over the tracked h600 evidence inputs.
- `h600_governance_run_plan.json`: compact governance run plan for the issue #4243 closure slice.
- `reconciliation_h600_13268_13273.json`: reconciler output linking both finalizers through the public
  `source_manifest.json`.
- `checklist_13268.json` and `checklist_13273.json`: machine-readable trace checklists.
- `checklist_13268.md` and `checklist_13273.md`: Markdown renderings of the trace checklists.
- `SHA256SUMS`: checksums for the compact files in this directory.

## Current Result

Passing links:

- retrieval source manifest includes jobs `13268` and `13273`;
- tracked compact evidence directory exists;
- h600 evidence checksums verify;
- finalizer manifests are present and carry durable pointers to the tracked compact evidence;
- reconciler bridge rows include both jobs through the public source manifest;
- spine-citable tracked pointer exists at
  `docs/context/evidence/issue_3810_h600_interpretation_2026-07/README.md`.

No fail-closed blockers remain in the regenerated checklists. The next empirical action is outside
this PR: run any private queue campaign follow-up without changing the diagnostic-only claim
boundary here.
