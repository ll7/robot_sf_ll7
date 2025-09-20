# Data Model: Test Performance Budget Feature

## Purpose
Conceptual (non-code) description of entities supporting per-test performance budgets and reporting. Avoids implementation specifics.

## Entities

### PerformanceBudgetPolicy
- Description: Defines soft and hard runtime thresholds and related environment overrides.
- Fields:
  - soft_threshold_seconds (number)
  - hard_timeout_seconds (number)
  - report_count (integer; top N slow tests)
  - relax_env_var (string; name of env var enabling relaxed mode)
  - enforce_env_var (string; name of env var elevating soft breaches to failures)
  - consecutive_breach_issue_policy (string; textual policy description)
- Invariants:
  - soft_threshold_seconds < hard_timeout_seconds

### TestRuntimeSample
- Description: Single observation of a test's wall-clock runtime.
- Fields:
  - test_identifier (string; canonical node id)
  - duration_seconds (number)
  - timestamp (datetime)
  - run_context (string; local, CI, etc.)

### SlowTestRecord
- Description: Aggregated summary for a slow test within a run.
- Fields:
  - test_identifier
  - duration_seconds
  - breach_type (enum: none, soft, hard)
  - guidance (string; human readable optimization suggestions)

### SlowTestReport
- Description: Summary grouping of slowest tests for a run.
- Fields:
  - generated_at (datetime)
  - policy_reference (PerformanceBudgetPolicy)
  - tests (list[SlowTestRecord])
  - max_duration_seconds (number)

### ScenarioMinimizationStrategy
- Description: Declarative description of how a test reduces input workload.
- Fields:
  - original_scenarios (integer)
  - minimized_scenarios (integer)
  - preserved_semantics (string; justification of sufficiency)

### ResumeSemanticsAssertion
- Description: Captures expectation for resume tests (no duplication).
- Fields:
  - expected_initial_episodes (integer)
  - expected_second_run_additional (integer; typically 0)
  - rationale (string)

## Relationships
- PerformanceBudgetPolicy informs generation of SlowTestReport.
- SlowTestReport aggregates SlowTestRecord derived from TestRuntimeSample.
- ScenarioMinimizationStrategy supports guidance field in SlowTestRecord.
- ResumeSemanticsAssertion referenced by tests implementing resume logic checks.

## State Transitions (Conceptual)
1. Collect raw runtimes → Create TestRuntimeSample entries.
2. Apply policy thresholds → Classify breach_type.
3. Select top N by duration → Build SlowTestReport.tests.
4. Emit report & integrate into test session output.

## Validation Rules
- All durations must be non-negative.
- breach_type is "hard" iff duration_seconds >= hard_timeout_seconds.
- soft breach occurs when soft_threshold_seconds <= duration_seconds < hard_timeout_seconds.

## Notes
This model is intentionally minimal; no persistence or external storage implied.
