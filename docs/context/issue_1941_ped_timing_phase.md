# Issue #1941 Pedestrian Timing Phase Perturbation

Issue: [#1941](https://github.com/ll7/robot_sf_ll7/issues/1941)
Parent: [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
Predecessors: [#1937](https://github.com/ll7/robot_sf_ll7/issues/1937),
[#1939](https://github.com/ll7/robot_sf_ll7/issues/1939)

## Goal

Add a narrowly bounded timing perturbation family to the #1610 scenario-perturbation pilot so
local planners can be probed for phase sensitivity in Francis single-pedestrian interactions.
This note is diagnostic local evidence only; it is not benchmark-strength or paper-facing evidence.

## Implementation

`single_pedestrian_start_delay_offset` changes `start_delay_s` on selected
`single_pedestrians` before certification and materialization. The family deliberately does not
target route-based pedestrian trajectories or `wait_at` waypoints; unsupported scenarios fail
closed and are excluded from success evidence.

The public manifest schema now distinguishes family-specific parameter requirements:

- route-offset families require `dx_m`, `dy_m`, and `max_magnitude_m`;
- start-delay offsets require `dt_s` and `max_abs_dt_s`;
- no-op variants must remain parameter-free.

The tracked pilot manifest is
`configs/scenarios/perturbations/issue_1610_ped_timing_phase_pilot_v1.yaml`.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
uv run python scripts/validation/run_scenario_perturbation_criticality_pilot.py \
  configs/scenarios/perturbations/issue_1610_ped_timing_phase_pilot_v1.yaml \
  --materialized-output-dir output/scenario_perturbations/issue1941_ped_timing_phase/materialized \
  --pilot-output-dir output/scenario_perturbations/issue1941_ped_timing_phase/pilot \
  --seed-limit 4 \
  --horizon 80 \
  --workers 1 \
  --planner goal \
  --planner orca \
  --planner scenario_adaptive_hybrid_orca_v2_collision_guard \
  --evidence-summary docs/context/evidence/issue_1941_ped_timing_phase_2026-05-31/summary.json
```

## Result

The diagnostic pilot completed 21/21 paired no-op-versus-start-delay rows with no invalid,
fallback, degraded, missing, or failed pair statuses. The route-only
`classic_head_on_corridor_low_start_delay_all_p050` probe failed closed during preflight and was
excluded from paired success evidence, as intended.

Mean completed-pair deltas over `goal`, `orca`, and
`scenario_adaptive_hybrid_orca_v2_collision_guard`:

| Slice | Pairs | Min-Distance Delta | Success Delta | Collision Delta | Timeout Delta |
|---|---:|---:|---:|---:|---:|
| all start-delay pairs | 21 | `+1.784229 m` | `0.0` | `0.0` | `0.0` |
| `francis2023_intersection_wait` | 9 | `+4.159358 m` | `0.0` | `0.0` | `0.0` |
| `francis2023_join_group` | 12 | `+0.002882 m` | `0.0` | `0.0` | `0.0` |
| `goal` | 7 | `+2.084558 m` | `0.0` | `0.0` | `0.0` |
| `orca` | 7 | `+1.595997 m` | `0.0` | `0.0` | `0.0` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 7 | `+1.672131 m` | `0.0` | `0.0` | `0.0` |

Interpretation: delaying the crossing pedestrian in `francis2023_intersection_wait` changes the
closest-approach geometry substantially in this short horizon, but it does not change success,
collision, or timeout outcomes. Delaying the joining pedestrian in `francis2023_join_group` barely
moves the min-distance metric. This is useful as a phase-sensitivity diagnostic and not evidence
that timing delays improve planner robustness.

## Evidence Boundary

Tracked compact evidence will live under
`docs/context/evidence/issue_1941_ped_timing_phase_2026-05-31/`:

- [summary.json](evidence/issue_1941_ped_timing_phase_2026-05-31/summary.json)
- [README.md](evidence/issue_1941_ped_timing_phase_2026-05-31/README.md)
- [SHA256SUMS](evidence/issue_1941_ped_timing_phase_2026-05-31/SHA256SUMS)

Ignored local outputs:

- materialized matrix and scenario overrides under
  `output/scenario_perturbations/issue1941_ped_timing_phase/`
- coverage output from targeted tests

The run emitted the known `uni_campus_big.svg` invalid obstacle warning during combined scenario
materialization but completed all eligible paired rows.

## Routing

This timing family is intentionally conservative. A later trajectory timing family can explore
pedestrian routes or `wait_at` phase perturbations, but that should be a separate issue because it
needs a different validity contract.
