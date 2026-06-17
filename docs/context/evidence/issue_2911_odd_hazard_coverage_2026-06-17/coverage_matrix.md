# ODD Hazard Coverage Matrix

- Matrix id: `issue_2911_low_speed_public_space_v1`
- Generated at: 2026-06-17T04:14:29.519494+00:00
- Generation command: `uv run python scripts/tools/generate_odd_hazard_coverage_matrix.py --config configs/benchmarks/odd_hazard_coverage.v1.yaml --out-json docs/context/evidence/issue_2911_odd_hazard_coverage_2026-06-17/coverage_matrix.json --out-md docs/context/evidence/issue_2911_odd_hazard_coverage_2026-06-17/coverage_matrix.md --repo-root .`
- Generation commit: `d7656d7f8494`

## Claim Boundary

This matrix records which ODD conditions and hazard classes are represented by checked-in configs only. It does not claim that any row has executed benchmark evidence, safety proof, or paper-facing validity. Uncovered or blocked rows must not be cited as covered in benchmark wording.


## Evidence Status Summary

- Coverage rows: 7
- Known gaps: 8
- Coverage status counts: weakly_covered=7
- Gap status counts: absent=2, blocked=6
- Reference validation: passed

## Coverage Rows

| ODD condition | Hazard class | Scenario family | Status | Evidence tier | Metrics | Planners | Gap reason |
|---|---|---|---|---|---|---|---|
| low_speed_public_space_v1 | robot_pedestrian_collision | station_platform | weakly_covered | diagnostic | collision_rate | goal; orca; social_force | Scenario contract exists, but no scenario_cert.v1 or executed benchmark evidence is checked in.  |
| low_speed_public_space_v1 | near_miss | classic_crossing | weakly_covered | candidate | min_ttc; pet | goal; orca; social_force; teb; dwa; risk_dwa; ppo | Included in paper_experiment_matrix_v1 checked-in config; durable runtime evidence is not committed.  |
| low_speed_public_space_v1 | blind_corner_emergence | blind_corner | weakly_covered | diagnostic | min_ttc; clearance | goal; orca; social_force | Smoke fixture exists; not integrated into a released benchmark campaign config with run evidence.  |
| low_speed_public_space_v1 | pedestrian_flow_disruption | corridor | weakly_covered | candidate | comfort_exposure_s; path_efficiency | goal; orca; social_force; teb; dwa; risk_dwa; ppo | Classic interactions matrix includes corridor family; no durable run evidence is committed.  |
| low_speed_public_space_v1 | robot_pedestrian_collision | bottleneck | weakly_covered | candidate | collision_rate | goal; orca; social_force; teb; dwa; risk_dwa; ppo | Classic interactions matrix includes bottleneck family; no durable run evidence is committed.  |
| low_speed_public_space_v1 | robot_pedestrian_collision | cross_trap | weakly_covered | diagnostic | collision_rate; near_miss | goal; orca; social_force | Scenario contract exists for #1484 rows; executed evidence is not checked in.  |
| low_speed_public_space_v1 | robot_pedestrian_collision | dense_pedestrian | weakly_covered | diagnostic | collision_rate; comfort_exposure_s | goal; orca; social_force | Dense-pedestrian stress fixture exists but is not a calibrated crowd benchmark.  |

## Known Gaps

| Gap id | Status | Affected hazard classes | Affected scenario families | Reason | Tracking issue |
|---|---|---|---|---|---|
| cyclist_vru_interaction | absent | robot_vru_collision | cyclist_interaction | No checked-in scenario contract or benchmark config exercises cyclist/VRU actors; Issue #2473 remains proposal only.  | https://github.com/ll7/robot_sf_ll7/issues/2473 |
| signalized_crossing | blocked | signal_violation; pedestrian_flow_disruption | signalized_crossing | Signalized crossing config (Issue #2474) and signal-state contract (Issue #2662) exist, but signal state is proxy-only and no planner-observable benchmark evidence is checked in.  | https://github.com/ll7/robot_sf_ll7/issues/2474 |
| occluded_emergence_benchmark | blocked | blind_corner_emergence | occluded_emergence | Occluded-emergence fixtures exist but are not integrated into a released benchmark matrix with executed run evidence.  | https://github.com/ll7/robot_sf_ll7/issues/2526 |
| stairs | absent | robot_fall_or_trip | stairs | No stair scenario, map, or contract is checked in.  | — |
| dense_crowds | blocked | pedestrian_flow_disruption | dense_pedestrian | Dense-pedestrian stress fixture is limited to a few deterministic pedestrians and is not a calibrated crowd benchmark. This gap is distinct from the weakly_covered dense_pedestrian row, which records only that a checked-in fixture exists.  | https://github.com/ll7/robot_sf_ll7/issues/1959 |
| narrow_lane_conflict | blocked | near_miss; robot_pedestrian_collision | doorway; t_intersection | Narrow passages appear in archetypes, but no explicit narrow-lane conflict scenario family is checked in.  | https://github.com/ll7/robot_sf_ll7/issues/1318 |
| amv_actuation_limits | blocked | amv_actuation_limit | amv_actuation | Synthetic AMV actuation stress slices exist (Issue #1556, Issue #2011) but are not hardware-calibrated or platform-class evidence.  | https://github.com/ll7/robot_sf_ll7/issues/2230 |
| sensor_latency | blocked | observation_delay | observation_noise | Observation-noise/latency smoke configs exist, but no systematic closed-loop benchmark with sensor-latency claims is checked in.  | https://github.com/ll7/robot_sf_ll7/issues/1744 |

## Benchmark Wording Guard

- Rows with status `covered` may be described as represented by checked-in configs only.
- Rows with status `weakly_covered` must be described as config-only, candidate, or diagnostic.
- Rows with status `blocked` or `absent` must not be described as covered, benchmark-success, or paper-facing evidence.
- This report is schema/proposal evidence unless a separate benchmark run updates the evidence tier.

## Provenance

- Source issue: https://github.com/ll7/robot_sf_ll7/issues/2911
- Authored by: robot_sf_ll7
- Source files:
  - configs/benchmarks/odd_hazard_coverage.v1.yaml
  - configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml
  - configs/benchmarks/hazard_traceability/low_speed_public_space_v1.yaml

Static coverage matrix for Issue #2911. All rows are metadata/candidate/diagnostic unless a separate benchmark run produces durable evidence and updates this matrix.

