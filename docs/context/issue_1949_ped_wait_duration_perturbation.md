# Issue #1949 Pedestrian Wait-Duration Perturbation

Issue: [#1949](https://github.com/ll7/robot_sf_ll7/issues/1949)
Parent: [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
Predecessors: [#1941](https://github.com/ll7/robot_sf_ll7/issues/1941),
[#1943](https://github.com/ll7/robot_sf_ll7/issues/1943),
[#1947](https://github.com/ll7/robot_sf_ll7/issues/1947)

## Goal

Add a narrowly bounded `single_pedestrian_wait_duration_offset` family to the #1610
scenario-perturbation pilot lane. This note is diagnostic local evidence only; it is not
benchmark-strength or paper-facing evidence.

## Implementation

`single_pedestrian_wait_duration_offset` changes selected explicit `single_pedestrians` by adding
`wait_delta_s` to each selected `wait_at[].wait_s` before certification and materialization. The
family is deliberately limited to explicit single-pedestrian wait rules:

- selected pedestrians without `wait_at` entries fail closed;
- route-only scenarios without explicit `single_pedestrians` fail closed;
- offsets that exceed `parameters.max_abs_wait_delta_s` or `validity.max_wait_duration_offset_s`
  fail before certification;
- offsets that would make any updated `wait_s` negative fail closed.

Materialized scenario overrides preserve existing `single_pedestrians` entries and update only the
selected `wait_at` payloads. The tracked pilot manifest is
`configs/scenarios/perturbations/issue_1610_ped_wait_duration_pilot_v1.yaml`.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
uv run python scripts/validation/run_scenario_perturbation_criticality_pilot.py \
  configs/scenarios/perturbations/issue_1610_ped_wait_duration_pilot_v1.yaml \
  --materialized-output-dir output/scenario_perturbations/issue1949_ped_wait_duration/materialized \
  --pilot-output-dir output/scenario_perturbations/issue1949_ped_wait_duration/pilot \
  --seed-limit 4 \
  --horizon 80 \
  --dt 0.1 \
  --workers 1 \
  --planner goal \
  --planner orca \
  --planner scenario_adaptive_hybrid_orca_v2_collision_guard \
  --evidence-summary docs/context/evidence/issue_1949_ped_wait_duration_perturbation_2026-06-01/summary.json
```

## Result

The diagnostic pilot completed 9/9 eligible no-op-versus-wait-duration rows with no invalid,
fallback, degraded, missing, or failed pair statuses. The `francis2023_join_group_wait_h3_p050`
probe failed closed because selected single pedestrians had no `wait_at` entries. The
`classic_head_on_corridor_low_wait_all_p050` probe failed closed because the route-only scenario
has no explicit `single_pedestrians`.

Mean completed-pair deltas over `goal`, `orca`, and
`scenario_adaptive_hybrid_orca_v2_collision_guard`:

| Slice | Pairs | Min-Distance Delta | Success Delta | Collision Delta | Timeout Delta |
|---|---:|---:|---:|---:|---:|
| all wait-duration pairs | 9 | `0.0 m` | `0.0` | `0.0` | `0.0` |
| `francis2023_intersection_wait` | 9 | `0.0 m` | `0.0` | `0.0` | `0.0` |
| `goal` | 3 | `0.0 m` | `0.0` | `0.0` | `0.0` |
| `orca` | 3 | `0.0 m` | `0.0` | `0.0` | `0.0` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 3 | `0.0 m` | `0.0` | `0.0` | `0.0` |

Interpretation: extending `francis2023_intersection_wait` pedestrian `h1` from a 2.0 s wait to a
2.5 s wait did not change the recorded short-horizon outcomes or closest-approach distances for
the three planners and three seeds tested. This is a useful negative diagnostic: on this exact
surface, the wait-duration offset did not reproduce the phase sensitivity observed for start delay
in #1941. It should not be read as evidence that wait duration is globally unimportant.

## Evidence Boundary

Tracked compact evidence:

- [summary.json](evidence/issue_1949_ped_wait_duration_perturbation_2026-06-01/summary.json)
- [README.md](evidence/issue_1949_ped_wait_duration_perturbation_2026-06-01/README.md)
- [SHA256SUMS](evidence/issue_1949_ped_wait_duration_perturbation_2026-06-01/SHA256SUMS)

Ignored local outputs:

- materialized matrix under
  `output/scenario_perturbations/issue1949_ped_wait_duration/materialized/`
- raw planner JSONL and local Markdown summary under
  `output/scenario_perturbations/issue1949_ped_wait_duration/pilot/`
- coverage output from targeted tests

The run emitted the known `uni_campus_big.svg` invalid obstacle warning during combined scenario
materialization but completed all eligible paired rows.

## Routing

The family is intentionally single-pedestrian wait-rule only. A broader dwell/phase perturbation
campaign should add more wait-bearing scenarios or a separate route-pedestrian wait contract before
making stronger planner robustness claims.
