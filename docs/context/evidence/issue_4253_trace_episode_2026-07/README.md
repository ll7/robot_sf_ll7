# Issue #4268 Trace-Level Doorway Episode

Plain-language summary: this directory contains one reproducible, seed-pinned doorway episode trace
for dissertation Chapter 7 figure work. It is a single illustrative episode and does not establish a
statistical benchmark or dissertation claim.

## Contents

- `trace_timeseries.csv`: per-timestep robot state, commanded action, executed velocity, pedestrian
  positions, and nearest robot-pedestrian distance.
- `min_distance_series.csv`: figure-ready `(step, time_s, min_robot_ped_distance_m)` series.
- `trace_series.json`: raw recorded frames plus derived rows.
- `metadata.json`: scenario, seed, planner, commit, matrix hash, and claim boundary.
- `SHA256SUMS`: checksums for the files above.

## Reproduction

```bash
LOGURU_LEVEL=WARNING uv run python scripts/export_issue_4268_trace_episode.py
```

Pinned run:

- scenario matrix: `configs/scenarios/classic_interactions.yaml`
- scenario id: `classic_doorway_medium`
- seed: `141`
- planner: `simple_policy`
- horizon: `100`
- dt: `0.1`
- git commit at generation: `022f60c69749861a578d1bc0a6ca87e0e39d2040`

## Claim Boundary

This bundle is `analysis_workbench_only` style evidence for one illustrative episode. It should be
used as a trace-level worked example input only. It is not a full benchmark campaign, not a Slurm or
GPU result, and not a statistical comparison.
