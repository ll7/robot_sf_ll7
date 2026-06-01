# Issue #1953 Intersection-Wait Speed-Grid Trace Evidence

This bundle preserves compact diagnostic-only evidence for issue #1953. It supports
`docs/context/issue_1953_intersection_wait_speed_grid_trace.md`.

## Contents

- `closest_approach_trace_slices_speed_h1_p050.json`: full compact trace-run output for the
  targeted `francis2023_intersection_wait_speed_h1_p050` row.
- `focus_speed_h1_p050_summary.json`: smaller focused JSON table with aggregate and per-pair
  closest-frame fields used by the context note.
- `report_speed_h1_p050.md`: runner-generated Markdown summary for the targeted `p050` run.
- `closest_approach_trace_slices.json`: literal-family command output. This selected
  `francis2023_intersection_wait_speed_h1_m025` because the current runner chooses the first
  matching family variant.
- `report.md`: runner-generated Markdown summary for the literal-family `m025` run.
- `SHA256SUMS`: checksums for the tracked evidence files in this bundle.

## Boundary

The evidence is diagnostic local trace inspection only. It is not benchmark-strength or
paper-facing evidence. Raw materialized scenario matrices remain ignored under `output/` and are
reproducible from the tracked #1951 manifest plus the commands recorded in the context note.
