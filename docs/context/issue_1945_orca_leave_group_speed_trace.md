# Issue #1945 ORCA Leave-Group Speed Trace

Issue: [#1945](https://github.com/ll7/robot_sf_ll7/issues/1945)
Parent: [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
Predecessor: [#1943](https://github.com/ll7/robot_sf_ll7/issues/1943)

## Goal

Inspect the trace-level mechanism behind the ORCA `francis2023_leave_group` seed `258`
collision-to-success flip observed in the single-pedestrian speed perturbation pilot. This note is
diagnostic local evidence only; it is not benchmark-strength or paper-facing evidence.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
uv run python scripts/validation/run_scenario_perturbation_trace_response.py \
  configs/scenarios/perturbations/issue_1610_ped_speed_pilot_v1.yaml \
  --materialized-output-dir output/scenario_perturbations/issue1945_orca_leave_group_speed_trace/materialized \
  --output docs/context/evidence/issue_1945_orca_leave_group_speed_trace_2026-06-01/closest_approach_trace_slices.json \
  --markdown-output docs/context/evidence/issue_1945_orca_leave_group_speed_trace_2026-06-01/report.md \
  --source-scenario-id francis2023_leave_group \
  --perturbed-family single_pedestrian_speed_offset \
  --seed-limit 4 \
  --horizon 80 \
  --dt 0.1 \
  --slice-window 3 \
  --planner orca
```

The run emitted the known `uni_campus_big.svg` invalid obstacle warning during combined scenario
materialization but completed all selected trace pairs.

## Result

- Source scenario: `francis2023_leave_group`.
- Perturbed family: `single_pedestrian_speed_offset`.
- Planner: `orca`.
- Pair rows: 3 completed pairs, 0 excluded pairs.
- Mean closest-approach deltas over completed pairs:
  - center distance: `+0.034504 m`
  - clearance: `+0.034504 m`
  - progress: `+0.930346 m`
  - closest-approach time: `-0.433333 s`

Seed `258`, the terminal-outcome flip from #1943, is the important row:

| Run | Termination | Closest Time | Closest Ped Index | Center Distance | Clearance | Robot Position | Ped Position |
|---|---|---:|---:|---:|---:|---|---|
| no-op | `collision` | `3.8 s` | 2 | `1.398979 m` | `-0.001021 m` | `[12.023047, 9.101211]` | `[12.0, 10.5]` |
| speed offset | `success` | `3.2 s` | 0 | `1.412227 m` | `+0.012227 m` | `[10.675458, 9.044221]` | `[10.769671, 10.453302]` |

Interpretation: the ORCA seed-258 outcome flip is not explained by a large clearance margin. The
no-op row clips the static group anchor by roughly one millimeter at closest approach and
terminates as collision. The speed-offset row shifts the closest-approach event about `0.6 s`
earlier, changes the closest pedestrian from the static anchor to the moving leaving pedestrian,
and remains just outside collision clearance before eventually reaching success. This is a fragile
phase/order mechanism, not evidence that speed perturbations robustly improve ORCA.

Neighboring seeds reinforce the seed-local caveat:

- Seed `259` remains collision in both rows; the speed row reaches closest approach `0.9 s` earlier
  and has slightly worse clearance (`-0.010947 m` delta).
- Seed `260` remains success in both rows; the speed row has a larger positive clearance delta
  (`+0.101211 m`) and closest approach occurs `0.2 s` later.

## Evidence Boundary

Tracked compact evidence:

- [closest_approach_trace_slices.json](evidence/issue_1945_orca_leave_group_speed_trace_2026-06-01/closest_approach_trace_slices.json)
- [report.md](evidence/issue_1945_orca_leave_group_speed_trace_2026-06-01/report.md)
- [README.md](evidence/issue_1945_orca_leave_group_speed_trace_2026-06-01/README.md)
- [SHA256SUMS](evidence/issue_1945_orca_leave_group_speed_trace_2026-06-01/SHA256SUMS)

Ignored local outputs:

- materialized matrix and scenario overrides under
  `output/scenario_perturbations/issue1945_orca_leave_group_speed_trace/`
- coverage output from targeted tests

## Routing

The next useful follow-up would be trace-level comparison of `francis2023_intersection_wait`
timing versus speed perturbations if the goal is to understand clearance sensitivity. Route-based
pedestrian speed perturbations should remain separate until a real route-speed contract exists.
