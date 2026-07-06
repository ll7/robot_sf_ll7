# Issue #2910 Publication Suite Certification Report

Status: `blocked`

Claim boundary: CPU-only integration report over tracked scenario_cert.v1 summary and release claim matrix. It does not run a benchmark campaign, publish a release, submit compute, or promote blocked/stress-only rows as benchmark evidence.

## Summary

- Scenario count: 48
- Benchmark eligibility: {'eligible': 37, 'excluded': 2, 'stress_only': 9}
- Release artifact rows blocked on certification: 3

## Blockers

- `excluded_scenarios` (2): Remove these geometrically infeasible scenarios from the v0.1 publication suite or repair the geometry and regenerate scenario_cert.v1 evidence.
- `stress_only_scenarios` (9): Route these scenarios through an explicit stress-suite claim boundary or keep them out of the nominal v0.1 publication set.
- `release_claim_matrix_rows` (3): Regenerate the release claim matrix after the publication suite exclusion/stress-only policy is applied.

## Excluded Scenarios

| Scenario | Classification | Reason |
| --- | --- | --- |
| `francis2023_exiting_elevator` | geometrically_infeasible | no_inflated_collision_free_path: Planning aborted start/goal is in invalid cell; resample start/goal retry.; last error: start at (12.25, 10.25) is in invalid cell (inflated area); choose free/start/goal cell. |
| `francis2023_narrow_doorway` | geometrically_infeasible | no_inflated_collision_free_path: Planning failed after trying inflation radii [2] |

## Stress-Only Scenarios

| Scenario | Classification | Reason |
| --- | --- | --- |
| `classic_cross_trap_high` | stress_only |  |
| `classic_cross_trap_low` | stress_only |  |
| `classic_cross_trap_medium` | stress_only |  |
| `classic_doorway_high` | stress_only |  |
| `classic_doorway_low` | stress_only |  |
| `classic_doorway_medium` | stress_only |  |
| `francis2023_entering_elevator` | stress_only |  |
| `francis2023_entering_room` | stress_only |  |
| `francis2023_exiting_room` | stress_only |  |

## Release Claim Matrix Rows Still Blocked

| Row | Classification | Scenario certification |
| --- | --- | --- |
| `release_artifact:tab_release_failure_count_slices` | benchmark evidence | scenario_cert.v1:blocked |
| `release_artifact:tab_results_overview` | benchmark evidence | scenario_cert.v1:blocked |
| `release_artifact:tab_robot_sf_release_planner_results` | benchmark evidence | scenario_cert.v1:blocked |

## Next Empirical Action

Apply a versioned v0.1 suite policy that excludes or repairs the 2 geometrically infeasible scenarios and explicitly routes or removes the 9 stress-only scenarios, then regenerate scenario_cert.v1 summary, release claim matrix, and publication gate output.
