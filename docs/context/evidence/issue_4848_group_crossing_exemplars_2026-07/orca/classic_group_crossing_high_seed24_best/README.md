<!-- AI-GENERATED (robot_sf#4848, 2026-07-08) - NEEDS-REVIEW -->

# Issue #4848 Exemplar Trace: classic_group_crossing_high (orca)

Plain-language summary: this directory contains one exemplar trace episode from the
retained `issue4206_trace_capable_h600_rerun_20260704` campaign (job 13334).
It is an illustrative group-crossing interaction episode and does not establish a
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
- Planner: `orca`
- Scenario: `classic_group_crossing_high`
- Seed: `24`
- Selection mode: `best`
- Selection metric: `path_efficiency = 1.0`
- Git commit at generation: `50c36e7e5eea05f6951c127019eee019bc2a8b70`

## Claim Boundary

This bundle is `illustrative_exemplar` evidence for one group-crossing episode.
It should be used for visualization and worked example input only. It is not a full
benchmark campaign, not a Slurm or GPU result, and not a statistical comparison.


<!-- /AI-GENERATED -->