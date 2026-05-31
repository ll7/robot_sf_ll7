# Issue #1939 Corridor Trace Response

Issue: [#1939](https://github.com/ll7/robot_sf_ll7/issues/1939)
Parent: [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
Predecessor: [#1937](https://github.com/ll7/robot_sf_ll7/issues/1937)

## Goal

Inspect the trace-level mechanism behind the larger `classic_head_on_corridor_low`
`pedestrian_route_offset` min-distance response observed in the #1937 pilot. This note is
diagnostic local evidence only; it is not benchmark-strength or paper-facing evidence.

## Implementation

`scripts/validation/run_scenario_perturbation_trace_response.py` materializes the tracked #1937
manifest, selects one no-op and one perturbed variant for a source scenario, reruns each selected
planner/seed pair, and records compact closest-approach slices instead of raw episode videos or
full traces.

The script resolves policy-search candidate keys through the same registry helper used by the
criticality pilot, so `scenario_adaptive_hybrid_orca_v2_collision_guard` remains a planner label
while executing its configured `hybrid_rule_local_planner` candidate config.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
uv run python scripts/validation/run_scenario_perturbation_trace_response.py \
  configs/scenarios/perturbations/issue_1610_ped_route_offset_pilot_v1.yaml \
  --materialized-output-dir output/scenario_perturbations/issue1939_corridor_trace_response/materialized \
  --output docs/context/evidence/issue_1939_corridor_trace_response_2026-05-31/closest_approach_trace_slices.json \
  --markdown-output docs/context/evidence/issue_1939_corridor_trace_response_2026-05-31/report.md \
  --source-scenario-id classic_head_on_corridor_low \
  --perturbed-family pedestrian_route_offset \
  --seed-limit 4 \
  --horizon 80 \
  --dt 0.1 \
  --slice-window 2 \
  --planner goal \
  --planner orca \
  --planner scenario_adaptive_hybrid_orca_v2_collision_guard
```

The run emitted the known `uni_campus_big.svg` invalid obstacle warning during materialization but
completed all selected trace pairs.

## Result

- Source scenario: `classic_head_on_corridor_low`.
- Perturbed family: `pedestrian_route_offset`.
- Planners: `goal`, `orca`, `scenario_adaptive_hybrid_orca_v2_collision_guard`.
- Pair rows: 12 completed pairs, 0 excluded pairs.
- Mean closest-approach deltas over completed pairs:
  - center distance: `+0.153489 m`
  - clearance: `+0.153489 m`
  - progress: `+0.506475 m`
  - closest-approach time: `-0.25 s`

By planner:

| Planner | Pairs | Center-Distance Delta | Progress Delta | Time Delta |
|---|---:|---:|---:|---:|
| `goal` | 4 | `+0.159909 m` | `-0.024578 m` | `+0.025 s` |
| `orca` | 4 | `+0.157236 m` | `-0.049732 m` | `+0.025 s` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 4 | `+0.143321 m` | `+1.593735 m` | `-0.8 s` |

Interpretation: the pedestrian-route offset generally moves the closest pedestrian farther from
the robot at closest approach in this corridor slice. The large hybrid-candidate progress delta is
driven by seed 117, where the perturbed trace reaches closest approach `3.3 s` earlier and roughly
`6.58 m` farther along the route than the no-op trace. Other hybrid seeds show clearance changes
without that route-progress jump, so the mechanism should be treated as seed-local diagnostic
evidence rather than a broad planner claim.

Seed 116 remains collision-terminated in both no-op and perturbed traces for all three planners.
The trace slices therefore explain clearance geometry around closest approach, not a solved
collision outcome.

## Evidence Boundary

Tracked compact evidence:

- [closest_approach_trace_slices.json](evidence/issue_1939_corridor_trace_response_2026-05-31/closest_approach_trace_slices.json)
- [report.md](evidence/issue_1939_corridor_trace_response_2026-05-31/report.md)
- [SHA256SUMS](evidence/issue_1939_corridor_trace_response_2026-05-31/SHA256SUMS)

Ignored local outputs:

- materialized matrix and route overrides under
  `output/scenario_perturbations/issue1939_corridor_trace_response/`
- coverage output from targeted tests

## Routing

Useful next #1610 children could test a timing or speed perturbation family, but the trace evidence
argues against presenting corridor pedestrian-route offsets as a success/robustness benchmark on
their own. They are currently a local diagnostic lens for clearance and route-progress mechanisms.
