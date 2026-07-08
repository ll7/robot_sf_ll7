<!-- AI-GENERATED (robot_sf#4891, 2026-07-09) - NEEDS-REVIEW -->

# Issue #4891 Exemplar Trace: classic_head_on_corridor_medium (social_force)

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
- Planner: `social_force`
- Scenario: `classic_head_on_corridor_medium`
- Seed: `24`
- Selection mode: `worst`
- Selection metric: `path_efficiency = 0.6836209993728016`
- Git commit at generation: `bb59e8b516a7fed1e014cc2d70b40af2478e747d`

## Claim Boundary

This bundle is `illustrative_exemplar` evidence for one head-on corridor episode.
It should be used for visualization and worked example input only. It is not a full
benchmark campaign, not a Slurm or GPU result, and not a statistical comparison.
