# Phase 0 Research: Accelerate Slow Benchmark Tests

## Purpose
Establish context, resolve open clarifications, and document decisions for enforcing per-test performance budgets (<20s soft, 60s hard) without reducing semantic coverage.

## Open Clarifications (from spec)
1. Top N slowest tests to report.
2. Enforcement action for repeated 20s+ offenders under 60s.
3. Environment variable name to relax soft threshold.

## Findings & Decisions
### 1. Top N Slowest Tests
- Candidate values: 5, 10, 15.
- Decision: Report top 10 for sufficient breadth without noise.
- Rationale: Aligns with common pytest --durations default patterns.
- Alternatives: 5 (risk missing borderline regressions), 15 (adds noise, harder triage).

### 2. Enforcement Action for 20s+ (soft breach)
- Options: (a) Warning only, (b) Mark xfail, (c) Fail build, (d) Require issue link.
- Decision: Warning + require adding a TODO with tracking issue if persists >2 consecutive PRs.
- Rationale: Avoids abrupt friction while enforcing accountability.
- Alternatives Rejected:
  * xfail: Masks regression and may accumulate.
  * Immediate fail: Too strict for transient CI variance.

### 3. Environment Variable Name
- Options: TEST_PERF_RELAX, ROBOT_SF_RELAX_SOFT_TIMING, ROBOT_SF_PERF_RELAX.
- Decision: Use ROBOT_SF_PERF_RELAX=1.
- Rationale: Namespace aligned with existing perf env vars (e.g., ROBOT_SF_PERF_CREATION_SOFT) promoting consistency.

### 4. Mechanism for Hard Timeout
- Use per-test timeout integration (pytest-timeout marker) consistent with existing pattern used in benchmark tests.
- Rationale: Minimal friction, no custom harness required.

### 5. Minimizing Scenario Matrices
- Strategy: Replace large scenario sets with a single canonical scenario that still passes through the logic under test (resume, reproducibility, config parsing).
- Ensure: Episode count deterministic (e.g., 2 episodes) and assertions verify count semantics.

### 6. Detecting Slow Tests
- Approach Options: (a) Rely on `pytest --durations`, (b) Custom plugin, (c) Post-processing JSON.
- Decision: Use pytest's built-in durations plus a thin parser in a helper to emit guidance if top test > soft threshold.
- Rationale: Avoid writing a plugin initially; incremental.

### 7. Advisory vs Hard Enforcement Implementation
- Advisory: On soft breach, print structured guidance with suggested reductions (scenario count, horizon, bootstrap samples, workers).
- Hard: Only when >60s or explicit env var (ROBOT_SF_PERF_ENFORCE=1) toggled.

### 8. Documentation Update Path
- Add subsection to `docs/dev_guide.md` under Testing strategy: "Per-Test Performance Budget".
- Link from central `docs/README.md`.

### 9. Change Log Entry
- Add under "Unreleased" section: Introduced per-test performance budget and slow test reporting.

### 10. Risk Mitigation for Flaky Timing
- Use wall clock measurement with monotonic timer; allow slack factor if ROBOT_SF_PERF_RELAX set.

## Resolved Clarifications
- FR-004 N=10.
- FR-008 Enforcement: Warning + tracking issue after 2 PRs.
- FR-009 Env var: ROBOT_SF_PERF_RELAX=1.

## Remaining Unknowns
- None (all prior NEEDS CLARIFICATION resolved).

## Decisions Summary Table
| Topic | Decision | Alternatives | Rationale |
|-------|----------|-------------|-----------|
| Slow report size | 10 | 5, 15 | Balance coverage vs noise |
| Soft breach handling | Warn + issue after 2 PRs | xfail, fail | Encourages timely action w/out friction |
| Relax env var | ROBOT_SF_PERF_RELAX | TEST_PERF_RELAX | Namespace consistency |
| Detection method | pytest durations | custom plugin | Leverage existing tooling |
| Hard timeout | pytest-timeout | custom wrapper | Simplicity |
| Scenario minimization | Single canonical scenario | Unmodified large set | Deterministic fast coverage |

## Next Steps (Gate to Phase 1)
Proceed to Phase 1: define conceptual data entities (PerformanceBudgetPolicy, SlowTestRecord) and contracts for reporting integration.
