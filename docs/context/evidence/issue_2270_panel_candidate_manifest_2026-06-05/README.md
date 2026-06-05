# Issue #2270 Panel Candidate Manifest

This directory preserves the compact analysis-only candidate manifest for Issue #2270 / parent
Issue #2227.

## Result

The repository has two useful mechanism-panel candidate pairs, but both remain blocked on durable
mechanism-specific `simulation_trace_export.v1` baseline/intervention trace exports:

- static-recentering held-out non-transfer on `classic_station_platform_medium`;
- topology primary-route-only behavior on `classic_realworld_double_bottleneck_high`.

## Tracked Files

- `panel_candidate_manifest.yaml`: candidate rows, source evidence, missing trace inputs, required
  exports, and recommended next command shapes.

## Claim Boundary

This is input-discovery evidence only. It is not a rendered panel bundle, not benchmark-strength
planner evidence, and not paper-facing mechanism proof.
