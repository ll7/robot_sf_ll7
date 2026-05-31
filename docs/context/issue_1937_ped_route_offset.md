# Issue #1937 Pedestrian Route Offset Pilot

Issue: [#1937](https://github.com/ll7/robot_sf_ll7/issues/1937)
Parent: [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
Predecessor: [#1935](https://github.com/ll7/robot_sf_ll7/issues/1935)

## Goal

Add exactly one new perturbation family, `pedestrian_route_offset`, to the #1610 paired
perturbation pilot while keeping the #1935 planner set, source scenarios, seed count, horizon, and
diagnostic-only evidence boundary fixed.

This is diagnostic local evidence only. It is not benchmark-strength or paper-facing evidence.

## Implementation

`scenario_perturbation_manifest.v1` now accepts `pedestrian_route_offset` with the same bounded
`dx_m`, `dy_m`, `max_magnitude_m`, and `waypoint_selector: all` parameter surface as
`robot_route_offset`.

The preflight and materializer share route-offset logic by family:

- `robot_route_offset` offsets selected `robot_routes` and writes only `robot_routes` in the route
  override payload.
- `pedestrian_route_offset` offsets selected `ped_routes` and writes only `ped_routes` in the route
  override payload.
- scenarios without pedestrian routes fail closed instead of being counted as executable evidence.

The issue #1937 pilot manifest retains the previous no-op and robot-route-offset variants, adds
pedestrian-route-offset variants for the same source scenarios, and intentionally records the
Francis join-group pedestrian-route row as excluded because that scenario uses single-pedestrian
circles rather than `ped_routes`.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
uv run python scripts/validation/run_scenario_perturbation_criticality_pilot.py \
  configs/scenarios/perturbations/issue_1610_ped_route_offset_pilot_v1.yaml \
  --materialized-output-dir output/scenario_perturbations/issue1937_ped_route_offset/materialized \
  --pilot-output-dir output/scenario_perturbations/issue1937_ped_route_offset/pilot \
  --seed-limit 4 \
  --horizon 80 \
  --workers 1 \
  --planner goal \
  --planner orca \
  --planner scenario_adaptive_hybrid_orca_v2_collision_guard \
  --evidence-summary docs/context/evidence/issue_1937_ped_route_offset_2026-05-31/summary.json
```

## Result

- Materialized variants: 8 included, 1 excluded.
- Excluded variant: `francis2023_join_group_ped_route_offset_x025`, because the source scenario has
  no pedestrian routes to offset.
- Planners: `goal`, `orca`, `scenario_adaptive_hybrid_orca_v2_collision_guard`.
- Pair rows: 60 completed pairs, 0 excluded pairs.
- Mean deltas over completed pairs:
  - success delta: `0.0000`
  - collision delta: `0.0000`
  - timeout delta: `0.0000`
  - min-distance delta: `+0.0461 m`

Grouped min-distance deltas:

| Group | Pairs | Mean Min-Distance Delta |
|---|---:|---:|
| `pedestrian_route_offset` | 24 | `+0.0978 m` |
| `robot_route_offset` | 36 | `+0.0116 m` |
| `goal` | 20 | `+0.0477 m` |
| `orca` | 20 | `+0.0295 m` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 20 | `+0.0610 m` |
| `classic_group_crossing_high` | 24 | `+0.0247 m` |
| `classic_head_on_corridor_low` | 24 | `+0.0886 m` |
| `francis2023_join_group` | 12 | `+0.0039 m` |

Interpretation: pedestrian-route offsets produce a noticeably larger clearance response than the
robot-route offsets on the route-based classic scenarios. The pilot remains neutral for success,
collision, and timeout, so it is useful planner/scenario diagnostic evidence but not a
benchmark-strength robustness failure claim.

## Evidence Boundary

Tracked compact evidence:

- [summary.json](evidence/issue_1937_ped_route_offset_2026-05-31/summary.json)
- [SHA256SUMS](evidence/issue_1937_ped_route_offset_2026-05-31/SHA256SUMS)

Ignored local outputs:

- materialized matrix and route overrides under `output/scenario_perturbations/issue1937_ped_route_offset/`
- raw planner episode JSONL
- local `summary.json` / `summary.md` generated beside the raw outputs

Fallback, degraded, invalid, missing, and failed rows are classified separately by the pilot script
and excluded from completed-pair means. This run had no such rows in the included executable pilot
matrix. The Francis pedestrian-route variant was excluded before execution by the fail-closed
preflight.

## Routing

The next #1610 child should not simply add more small offsets. More useful next steps are a
trace-level inspection of why the corridor pedestrian-route offset improves minimum distance, or a
new perturbation family that affects timing/phase while preserving the same diagnostic boundary.
