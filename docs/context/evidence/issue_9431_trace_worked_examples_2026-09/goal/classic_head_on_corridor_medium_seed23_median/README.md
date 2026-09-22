<!-- AI-GENERATED (robot_sf#4891, 2026-09-22) - NEEDS-REVIEW -->
# Issue #4891 Exemplar Trace: classic_head_on_corridor_medium (goal)

Plain-language summary: this directory contains one exemplar trace episode from the
retained `issue9431_worked_traces_0_0_7_20260922` campaign (job 15716).
It is an illustrative head-on corridor interaction episode and does not establish a
statistical benchmark or dissertation claim.

## Contents

- `trace_timeseries.csv`: per-timestep robot state, commanded action, executed velocity,
  pedestrian positions, and nearest robot-pedestrian distance.
- `min_distance_series.csv`: figure-ready `(step, time_s, min_robot_ped_distance_m)` series.
  Distance convention: `center_center` (robot-to-pedestrian center distance; footprint radii
  are NOT subtracted). Do not read these values as surface clearance.
- `trace_series.json`: raw recorded frames plus derived rows.
- `metadata.json`: provenance, selection criteria, and claim boundary.
- `SHA256SUMS`: checksums for the files above.

## Provenance

- Campaign: `issue9431_worked_traces_0_0_7_20260922`
- Job: `15716`
- Planner: `goal`
- Scenario: `classic_head_on_corridor_medium`
- Seed: `23`
- Selection mode: `median`
- Selection metric: `path_efficiency = 1.0`
- Git commit at generation: `07f7e8d43084de748915e1b1eb8b2a1603357c6e`
- Source commit: `07f7e8d43084de748915e1b1eb8b2a1603357c6e`
- Release tag: `paper-matrix-v2-h600-s30-2026-09-07f7e8d43084de748915e1b1eb8b2a1603357c6e`
- Config SHA-256: `dcaf87a46182e1cb64283a7b2442b02ce5fec86d33e9d72df2bcc6a04ee19806`

## Claim Boundary

This bundle is `illustrative_exemplar` evidence for one head-on corridor episode.
It should be used for visualization and worked example input only. It is not a full
benchmark campaign, not a Slurm or GPU result, and not a statistical comparison.

<!-- /AI-GENERATED -->
