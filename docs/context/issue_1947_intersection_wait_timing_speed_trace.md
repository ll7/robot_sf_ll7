# Issue #1947 Intersection-Wait Timing Vs Speed Trace

Issue: [#1947](https://github.com/ll7/robot_sf_ll7/issues/1947)
Parent: [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
Predecessors: [#1941](https://github.com/ll7/robot_sf_ll7/issues/1941),
[#1943](https://github.com/ll7/robot_sf_ll7/issues/1943), and
[#1945](https://github.com/ll7/robot_sf_ll7/issues/1945)

## Goal

Compare the trace-level response of `francis2023_intersection_wait` to the two single-pedestrian
perturbation families that produced opposite closest-clearance signals in the #1610 diagnostic
pilots:

- `single_pedestrian_start_delay_offset` from #1941.
- `single_pedestrian_speed_offset` from #1943.

This note is diagnostic local evidence only. It is not benchmark-strength or paper-facing evidence.

## Commands

Timing phase trace:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
uv run python scripts/validation/run_scenario_perturbation_trace_response.py \
  configs/scenarios/perturbations/issue_1610_ped_timing_phase_pilot_v1.yaml \
  --materialized-output-dir output/scenario_perturbations/issue1947_intersection_wait_timing_trace/materialized \
  --output docs/context/evidence/issue_1947_intersection_wait_timing_speed_trace_2026-06-01/timing_closest_approach_trace_slices.json \
  --markdown-output docs/context/evidence/issue_1947_intersection_wait_timing_speed_trace_2026-06-01/timing_report.md \
  --source-scenario-id francis2023_intersection_wait \
  --perturbed-family single_pedestrian_start_delay_offset \
  --seed-limit 4 \
  --horizon 80 \
  --dt 0.1 \
  --slice-window 3 \
  --planner goal --planner orca --planner scenario_adaptive_hybrid_orca_v2_collision_guard
```

Speed offset trace:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
uv run python scripts/validation/run_scenario_perturbation_trace_response.py \
  configs/scenarios/perturbations/issue_1610_ped_speed_pilot_v1.yaml \
  --materialized-output-dir output/scenario_perturbations/issue1947_intersection_wait_speed_trace/materialized \
  --output docs/context/evidence/issue_1947_intersection_wait_timing_speed_trace_2026-06-01/speed_closest_approach_trace_slices.json \
  --markdown-output docs/context/evidence/issue_1947_intersection_wait_timing_speed_trace_2026-06-01/speed_report.md \
  --source-scenario-id francis2023_intersection_wait \
  --perturbed-family single_pedestrian_speed_offset \
  --seed-limit 4 \
  --horizon 80 \
  --dt 0.1 \
  --slice-window 3 \
  --planner goal --planner orca --planner scenario_adaptive_hybrid_orca_v2_collision_guard
```

Both runs emitted the known `uni_campus_big.svg` invalid obstacle warning during combined scenario
materialization and completed all selected trace pairs.

## Result

Both families ran 9 completed pairs: 3 planners times 3 seeds. All rows kept the same termination
class (`max_steps -> max_steps`), so the comparison is about trace geometry and phase rather than
terminal outcome.

| Perturbation | Mean Clearance Delta | Mean Progress Delta | Mean Closest-Time Delta |
|---|---:|---:|---:|
| `single_pedestrian_start_delay_offset` | `+4.159358 m` | `+1.95458 m` | `-0.977778 s` |
| `single_pedestrian_speed_offset` | `-2.002917 m` | `-0.022215 m` | `+0.011111 s` |

Positive clearance/center-distance deltas mean the perturbed run was farther from the closest
pedestrian at closest approach.

By planner:

| Perturbation | Planner | Clearance Delta | Progress Delta | Closest-Time Delta |
|---|---|---:|---:|---:|
| timing | `goal` | `+4.861856 m` | `0.0 m` | `0.0 s` |
| timing | `orca` | `+3.714579 m` | `+2.998506 m` | `-1.5 s` |
| timing | `scenario_adaptive_hybrid_orca_v2_collision_guard` | `+3.901639 m` | `+2.865234 m` | `-1.433333 s` |
| speed | `goal` | `-2.298689 m` | `0.0 m` | `0.0 s` |
| speed | `orca` | `-1.820168 m` | `0.0 m` | `0.0 s` |
| speed | `scenario_adaptive_hybrid_orca_v2_collision_guard` | `-1.889893 m` | `-0.066644 m` | `+0.033333 s` |

## Interpretation

The opposite signs are consistent and visible across planners and seeds:

- The start-delay perturbation holds pedestrian `0` at the intersection longer. Closest approach
  still points to pedestrian `0`, but the delayed pedestrian is far ahead in phase space when the
  robot reaches the crossing. ORCA and the hybrid guard also move the robot roughly `2.9-3.0 m`
  farther along its route at an earlier closest-approach time.
- The speed-offset perturbation moves the same pedestrian along its route faster. The closest
  pedestrian remains pedestrian `0`, but closest clearance gets smaller by about `1.8-2.3 m`
  without meaningful route-progress or terminal-outcome changes.

This makes the `intersection_wait` signal a phase-sensitive crossing interaction: delaying the
intersecting pedestrian removes or postpones the conflict window, while increasing that
pedestrian's speed moves the pedestrian into a closer crossing phase. The trace does not justify a
planner robustness claim because no terminal outcome changed and only one scenario with three seeds
was inspected.

## Evidence Boundary

Tracked compact evidence:

- [timing_closest_approach_trace_slices.json](evidence/issue_1947_intersection_wait_timing_speed_trace_2026-06-01/timing_closest_approach_trace_slices.json)
- [timing_report.md](evidence/issue_1947_intersection_wait_timing_speed_trace_2026-06-01/timing_report.md)
- [speed_closest_approach_trace_slices.json](evidence/issue_1947_intersection_wait_timing_speed_trace_2026-06-01/speed_closest_approach_trace_slices.json)
- [speed_report.md](evidence/issue_1947_intersection_wait_timing_speed_trace_2026-06-01/speed_report.md)
- [README.md](evidence/issue_1947_intersection_wait_timing_speed_trace_2026-06-01/README.md)
- [SHA256SUMS](evidence/issue_1947_intersection_wait_timing_speed_trace_2026-06-01/SHA256SUMS)

Ignored local outputs:

- `output/scenario_perturbations/issue1947_intersection_wait_timing_trace/`
- `output/scenario_perturbations/issue1947_intersection_wait_speed_trace/`

## Routing

The next useful local-planner direction is to turn this phase sensitivity into a bounded search
policy probe: vary intersecting-pedestrian phase/speed over a small grid and rank planner responses
by closest-clearance and route-progress deltas, still under the diagnostic-only boundary until a
larger benchmark proof exists.
