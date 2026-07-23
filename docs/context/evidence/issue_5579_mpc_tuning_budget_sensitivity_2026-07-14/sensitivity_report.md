# Issue #5579 MPC Tuning-Budget Sensitivity

- Status: `blocked`
- Evidence tier: `diagnostic-only`
- Claim boundary: Diagnostic bounded sensitivity contract. A completed run can describe the tested fixed scenario slice and the observed best-found configuration, but it cannot establish a benchmark ranking, structural planner superiority, or a paper/dissertation claim. Fallback, degraded, failed, and unavailable rows are excluded from the success comparison and keep the report blocked.
- Execution-start commit: `b58316cbcd43eba7b9d7f61c323c149aae230fba`
- Config: `configs/analysis/issue_5579_mpc_tuning_sensitivity.yaml`
- Rows: 396 total, 295 eligible, 101 excluded (92 solver-failure and 9 fallback rows)

## Best-found target configurations

| Arm | Best candidate | Success rate | Eligible episodes | Excluded episodes |
| --- | --- | ---: | ---: | ---: |
| `prediction_mpc` | `horizon_high` | 0.000000 | 7 | 2 |
| `prediction_mpc_cbf` | `horizon_high` | 0.000000 | 7 | 2 |

## Incumbent hybrid band

| Arm | Success rate | Eligible episodes | Excluded episodes |
| --- | ---: | ---: | ---: |
| `scenario_adaptive_hybrid_orca_v1` | 0.000000 | 7 | 2 |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 0.000000 | 7 | 2 |
| `hybrid_rule_v3_fast_progress_static_escape` | 0.000000 | 7 | 2 |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | 0.000000 | 6 | 3 |

## Pre-registered read

- Decision: `blocked`
- Detail: Complete native/adapter rows are required before the pre-registered read.

All eligible target and incumbent summaries have zero observed successes. Because 101 rows were
excluded, those zeros do not distinguish a tuning-budget limitation from a structural limitation.

Fallback, degraded, failed, and unavailable rows are never treated as success evidence.
This diagnostic does not change benchmark metrics, roster status, or paper-facing claims.
