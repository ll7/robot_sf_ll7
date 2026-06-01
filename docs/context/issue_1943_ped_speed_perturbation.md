# Issue #1943 Single-Pedestrian Speed Perturbation

Issue: [#1943](https://github.com/ll7/robot_sf_ll7/issues/1943)
Parent: [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
Predecessors: [#1937](https://github.com/ll7/robot_sf_ll7/issues/1937),
[#1939](https://github.com/ll7/robot_sf_ll7/issues/1939),
[#1941](https://github.com/ll7/robot_sf_ll7/issues/1941)

## Goal

Add a narrowly bounded single-pedestrian speed perturbation family to the #1610
scenario-perturbation pilot lane. This note is diagnostic local evidence only; it is not
benchmark-strength or paper-facing evidence.

## Implementation

`single_pedestrian_speed_offset` changes selected `single_pedestrians` by adding
`speed_delta_m_s` to their baseline `speed_m_s` before certification and materialization. When a
source pedestrian omits `speed_m_s`, the perturbation uses the current runtime default single-ped
initial speed of `0.5 m/s` as the diagnostic baseline and writes an explicit `speed_m_s` override
into the materialized scenario.

The family deliberately does not target route-based pedestrians. `GlobalRoute` has no per-route
speed field, so route-only scenarios fail closed and are excluded from success evidence rather than
falling back to global `peds_speed_mult`.

The tracked pilot manifest is
`configs/scenarios/perturbations/issue_1610_ped_speed_pilot_v1.yaml`.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
uv run python scripts/validation/run_scenario_perturbation_criticality_pilot.py \
  configs/scenarios/perturbations/issue_1610_ped_speed_pilot_v1.yaml \
  --materialized-output-dir output/scenario_perturbations/issue1943_ped_speed/materialized \
  --pilot-output-dir output/scenario_perturbations/issue1943_ped_speed/pilot \
  --seed-limit 4 \
  --horizon 80 \
  --workers 1 \
  --planner goal \
  --planner orca \
  --planner scenario_adaptive_hybrid_orca_v2_collision_guard \
  --evidence-summary docs/context/evidence/issue_1943_ped_speed_perturbation_2026-05-31/summary.json
```

## Result

The diagnostic pilot completed 30/30 eligible no-op-versus-speed rows with no invalid, fallback,
degraded, missing, or failed pair statuses. The route-only
`classic_head_on_corridor_low_speed_all_p025` probe failed closed during preflight and was
excluded from paired success evidence, as intended.

Mean completed-pair deltas over `goal`, `orca`, and
`scenario_adaptive_hybrid_orca_v2_collision_guard`:

| Slice | Pairs | Min-Distance Delta | Success Delta | Collision Delta | Timeout Delta |
|---|---:|---:|---:|---:|---:|
| all speed pairs | 30 | `-0.588347 m` | `+0.033333` | `-0.033333` | `0.0` |
| `francis2023_intersection_wait` | 9 | `-2.002917 m` | `0.0` | `0.0` | `0.0` |
| `francis2023_join_group` | 12 | `-0.002172 m` | `0.0` | `0.0` | `0.0` |
| `francis2023_leave_group` | 9 | `+0.044654 m` | `+0.111111` | `-0.111111` | `0.0` |
| `goal` | 10 | `-0.695806 m` | `0.0` | `0.0` | `0.0` |
| `orca` | 10 | `-0.538939 m` | `+0.1` | `-0.1` | `0.0` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 10 | `-0.530298 m` | `0.0` | `0.0` | `0.0` |

Interpretation: increasing selected single-pedestrian speed generally decreases closest approach
distance in this short-horizon diagnostic, especially in `francis2023_intersection_wait`. The one
outcome change is ORCA on `francis2023_leave_group` seed 258, where the perturbed row changes from
collision to success while min distance changes only `+0.013248 m`. Treat that as a seed-local
mechanism clue, not a planner robustness claim.

## Evidence Boundary

Tracked compact evidence:

- [summary.json](evidence/issue_1943_ped_speed_perturbation_2026-05-31/summary.json)
- [README.md](evidence/issue_1943_ped_speed_perturbation_2026-05-31/README.md)
- [SHA256SUMS](evidence/issue_1943_ped_speed_perturbation_2026-05-31/SHA256SUMS)

Ignored local outputs:

- materialized matrix and scenario overrides under
  `output/scenario_perturbations/issue1943_ped_speed/`
- coverage output from targeted tests

The run emitted the known `uni_campus_big.svg` invalid obstacle warning during combined scenario
materialization but completed all eligible paired rows.

## Routing

This family is intentionally single-pedestrian-first. A later route-pedestrian speed perturbation
should define an explicit route-speed adapter or global-speed-multiplier contract before it is
allowed into the paired evidence lane.
