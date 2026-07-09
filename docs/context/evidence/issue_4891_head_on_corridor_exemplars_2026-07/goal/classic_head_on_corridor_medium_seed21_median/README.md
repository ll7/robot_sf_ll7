<!-- AI-GENERATED (robot_sf#4891, 2026-07-09) - NEEDS-REVIEW -->
# Issue #4891 Exemplar Trace: classic_head_on_corridor_medium (goal)

Plain-language summary: this directory contains one exemplar trace episode from the
retained `issue4206_trace_capable_h600_rerun_20260704` campaign (job 13334).
It is an illustrative head-on corridor interaction episode and does not establish a
statistical benchmark or dissertation claim.

## Contents

- `trace_timeseries.csv`: per-timestep robot state, commanded action, executed velocity,
  pedestrian positions, and nearest robot-pedestrian distance.
- `min_distance_series.csv`: figure-ready `(step, time_s, min_robot_ped_distance_m)` series.
- `trace_series.json`: raw recorded frames plus derived rows.
- `metadata.json`: provenance, selection criteria, and claim boundary.
- `SHA256SUMS`: checksums for the files above.

## Provenance

- Campaign: `issue4206_trace_capable_h600_rerun_20260704`
- Job: `13334`
- Planner: `goal`
- Scenario: `classic_head_on_corridor_medium`
- Seed: `21`
- Selection mode: `median`
- Selection metric: `path_efficiency = 1.0`
- Git commit at generation: `62e08add1c2a5a750a6fd68a823319280c2dccd6`

## Claim Boundary

This bundle is `illustrative_exemplar` evidence for one head-on corridor episode.
It should be used for visualization and worked example input only. It is not a full
benchmark campaign, not a Slurm or GPU result, and not a statistical comparison.

<!-- /AI-GENERATED -->
