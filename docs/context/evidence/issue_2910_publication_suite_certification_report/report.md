<!-- AI-GENERATED (robot_sf#2910, 2026-07-08) - NEEDS-REVIEW -->

# Issue #2910 Publication Suite Certification Report

Status: `blocked_pending_rebase`

Claim boundary: CPU-only integration report over tracked scenario_cert.v1 summary and release claim matrix. It does not run a benchmark campaign, publish a release, submit compute, or promote blocked/stress-only rows as benchmark evidence.

## Summary

- Scenario count: 48
- Benchmark eligibility: {'eligible': 37, 'excluded': 2, 'stress_only': 9}
- Release artifact rows blocked on certification: 0
- Publication-suite policy: `applied` (configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1_suite_policy.yaml)

## Blockers

- None.

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

## Next Empirical Action

Publication-suite policy ratified for 2 excluded and 9 stress-only scenarios; badge deferred pending #4364 release re-base. Regenerate matrix after re-base to flip badge.

<!-- /AI-GENERATED -->
