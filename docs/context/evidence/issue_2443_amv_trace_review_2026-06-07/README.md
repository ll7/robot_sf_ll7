# Issue #2443 AMV Trace Review Evidence

This directory contains the compact analysis-only evidence bundle for issue #2443.

Source artifacts:

- `docs/context/evidence/issue_2308_amv_timeout_trace_2026-06-05/summary.json`
- `docs/context/evidence/issue_2404_amv_timeout_decomposition_2026-06-06/summary.json`

Files:

- `summary.json`: compact review payload with candidate IDs, field availability, classification,
  matched-row deltas, and missing-field blockers.
- `progress_clipping_timeline.csv`: four 20-step progress/clipping windows for the baseline and
  actuation-aware intervention.

Boundary: this is diagnostic analysis over tracked compact summaries. It is not raw
`simulation_trace_export.v1`, benchmark-strength evidence, calibrated AMV evidence, hardware
evidence, planner-ranking evidence, or paper-facing evidence.
