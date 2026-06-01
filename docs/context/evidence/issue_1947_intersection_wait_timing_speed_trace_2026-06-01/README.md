# Issue #1947 Intersection-Wait Timing Vs Speed Trace Evidence

This bundle preserves compact diagnostic evidence for issue
[#1947](https://github.com/ll7/robot_sf_ll7/issues/1947). It compares
`francis2023_intersection_wait` closest-approach trace responses for:

- `single_pedestrian_start_delay_offset`
- `single_pedestrian_speed_offset`

The evidence is diagnostic local trace inspection only, not benchmark-strength or paper-facing
evidence.

## Contents

- `timing_closest_approach_trace_slices.json`: closest-approach trace slices for the timing
  perturbation family.
- `timing_report.md`: generated Markdown summary for the timing trace.
- `speed_closest_approach_trace_slices.json`: closest-approach trace slices for the speed
  perturbation family.
- `speed_report.md`: generated Markdown summary for the speed trace.
- `SHA256SUMS`: checksums for the tracked evidence files.

The materialized scenario matrices and overrides remain under ignored `output/` paths and are not
part of this tracked evidence bundle.
