# Issue #1500 Row Classification Report

Date: 2026-05-26

This manifest is a specification artifact, not benchmark evidence. The table below records how
future campaign rows must be interpreted.

| Row Type | Benchmark Evidence | Archive Eligible | Evidence Class | Readiness Status | Availability Status | Treatment |
|---|---:|---:|---|---|---|---|
| `valid_behavioral_failure` | no | yes | `development_stress_test` | `native` or `adapter` | available | stress-test archive row |
| `success` | no | no | `budget_audit_only` | `native` or `adapter` | available | budget audit row |
| `invalid_candidate` | no | no | `budget_audit_only` | not simulation evidence | not simulation evidence | explicit exclusion |
| `simulation_error` | no | no | `budget_audit_only` | `failed` | `failed` | explicit exclusion |
| `fallback` | no | no | `not_benchmark_evidence` | `fallback` | `not_available` | explicit caveat |
| `degraded` | no | no | `not_benchmark_evidence` | `degraded` | `not_available` | explicit caveat |
| `not_available` | no | no | `exclusion` | not applicable | `not_available` | explicit exclusion |

Fallback and degraded rows follow the fail-closed benchmark policy from
`docs/context/issue_691_benchmark_fallback_policy.md`: they are caveats, not successful
benchmark outcomes.
