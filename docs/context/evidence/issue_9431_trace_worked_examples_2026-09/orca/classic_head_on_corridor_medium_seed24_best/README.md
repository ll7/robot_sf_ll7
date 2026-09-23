<!-- AI-GENERATED (robot_sf#9431, 2026-09-23) - NEEDS-REVIEW -->
# Issue #9431 Exemplar Trace: classic_head_on_corridor_medium (orca)

Plain-language summary: this directory contains one exemplar trace episode from the
retained `issue9431_trace_18ep_0_0_7_2e87f9de0523_20260922` campaign (job 15724).
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

- Campaign: `issue9431_trace_18ep_0_0_7_2e87f9de0523_20260922`
- Job: `15724`
- Planner: `orca`
- Scenario: `classic_head_on_corridor_medium`
- Seed: `24`
- Selection mode: `best`
- Selection metric: `path_efficiency = 1.0`
- Git commit at generation: `f39bbcc7d49026c490d5cbda8bb1bdcfb18f8233`
- Source commit: `2e87f9de0523ef977bce19b40189ea0a74c61e13`
- Release tag: `paper-matrix-v2-h600-s30-2026-09-07f7e8d43084de748915e1b1eb8b2a1603357c6e`
- Config SHA-256: `9d0ddf20842e1d959a1f7428a4efb998ccab48e33d992b51bceeabb5781d90b5`

## Claim Boundary

This bundle is `illustrative_exemplar` evidence for one head-on corridor episode.
It should be used for visualization and worked example input only. It is not a full
benchmark campaign, not a Slurm or GPU result, and not a statistical comparison.

<!-- /AI-GENERATED -->
