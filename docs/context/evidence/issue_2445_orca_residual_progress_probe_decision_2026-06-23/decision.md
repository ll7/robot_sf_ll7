# Issue #2445 ORCA-Residual Progress-Probe Decision

- **Decision**: `stop`
- **Evidence Grade**: `analysis_only`
- **Prior decision issue**: #2408
- **Input artifact**: `docs/context/evidence/issue_1475_orca_residual_bc_smoke_12913_2026-06-17/summary.json`
- **Git HEAD**: `fedee58afe8c5834f4e6ccb1179cdc00a8354606`

## V1 Smoke Facts (job 12913)

| Field | Value |
| --- | --- |
| v1_smoke_success_rate | `0.0` |
| v1_smoke_failure_mode | `timeout_low_progress` |
| artifact_pointer_status | `missing` |
| missing_required_fields | `['residual_clipping_rate', 'guard_veto_rate', 'fallback_degraded_status', 'artifact_pointer_status']` |

## Rationale

Fail-closed: the v1 smoke is missing required smoke-evidence fields ['residual_clipping_rate', 'guard_veto_rate', 'fallback_degraded_status', 'artifact_pointer_status'] (summary status='failed_closed', nominal_escalation_allowed=False). Missing required fields independently forbid nominal escalation; combined with success_rate=0.0 and failure_mode='timeout_low_progress', the #2408 stop rule applies. Reopen only via a named objective/dataset redesign.

## Reopen Condition

Reopen only through a named objective/dataset redesign (a new BC objective, dataset, scenario, or instrumentation lane); do not rerun unchanged v0 BC and do not submit nominal_sanity from this failed-closed smoke.

## Claim Boundary

This is an analysis-only routing decision over existing failed-closed smoke evidence. It is not a benchmark result, not a learned-component success claim, and must not justify nominal or larger SLURM reruns.