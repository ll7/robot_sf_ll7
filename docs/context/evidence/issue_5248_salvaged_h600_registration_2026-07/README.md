<!-- AI-GENERATED (robot_sf#5248, 2026-07-14) - NEEDS-REVIEW -->
# Salvaged h600 harvest registration

Evidence status: `diagnostic-only`; registration is `blocked_campaign_registration`.

This packet replaces the placeholder blocked receipt from #5277. That earlier receipt was blocked
only because the verified job-13334 harvest was absent from the execution host. This slice runs the
same fail-closed checker (#5256) on `imech156-u`, where the salvaged harvest is present and
verified, and preserves the real receipt.

## What the checker actually observed

The campaign source artifacts are present and structurally valid:

- `reports/campaign_summary.json` read; `campaign.total_episodes == 6480` and
  `campaign.campaign_execution_status == "completed"`.
- `reports/seed_episode_rows.csv` read; exactly 6,480 data rows.
- Both source files are recorded with SHA-256 sums in `registration.json`.

The job was salvaged after the exit-code conflation fixed by #5240; the completed 6,480-episode
result confirms the FAILED mislabel was spurious. That provenance does **not** promote this packet
to benchmark, planner, paper, or dissertation evidence.

## Why registration stays blocked

The blocker is now the real one, not a missing-input artifact:

- All 6,480 episode rows carry `mechanism_label == "unknown"`,
  `mechanism_confidence == "unknown"`, `mechanism_evidence_mode == "unknown"`, and an empty
  `mechanism_evidence_uri`.
- Trace-verified labeled fraction is `0.000`, below the preregistered minimum `0.500`.
- This matches the root cause PR #4341 proved for the predecessor runs (jobs 13268 / 13273): the
  episode rows predate the trace-capable exporter (#4301), so the campaign completed but never
  captured trace-verified failure-mechanism labels. The preregistration contract's
  `all_not_derivable_output_is_success: false` clause requires this to fail closed.

## Reproduction

```bash
uv run python scripts/validation/check_issue_5248_salvaged_trace_rerun.py \
  --campaign-root output/issue4206-13334-harvest/issue4206_trace_capable_h600_rerun_20260704 \
  --job-id 13334 \
  --expected-total-episodes 6480 \
  --preregistration-config \
    configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml \
  --output-dir docs/context/evidence/issue_5248_salvaged_h600_registration_2026-07 \
  --generated-at 2026-07-14T175754Z
```

Expected result: exit code `2` and status `blocked_campaign_registration` (now from the trace-label
floor, not from missing inputs).

## Next action

Issue #4206's mechanism cross-cut cannot run against this job: its rows lack trace-verified labels.
That gap is its own instrumentation issue (trace capture #4301 was not enabled when job 13334 ran).
Do not substitute geometry buckets for mechanism labels (preregistration forbids it). The conditional
#4206 cross-cut remains blocked_pending_trace_verified_labels.
