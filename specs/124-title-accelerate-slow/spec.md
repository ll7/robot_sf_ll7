# Feature Specification: Accelerate Slow Benchmark Tests (Per-Test Performance Budget)

**Feature Branch**: `124-title-accelerate-slow`  
**Created**: 2025-09-20  
**Status**: Draft  
**Input Summary**: Improve test suite performance so no single test exceeds 20 seconds wall time locally; introduce systematic detection, prevention, and remediation of overly slow tests (notably long-running benchmark/"full classic" style tests).

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a maintainer running the test suite locally or in CI, I want each individual test to finish quickly (well under 20 seconds) so feedback cycles remain fast, contributing is frictionless, and performance regressions in tests are surfaced early.

### Acceptance Scenarios
1. **Given** a previously slow benchmark test that exceeded 60 seconds, **When** the maintainer runs the test after this feature, **Then** it completes in under 20 seconds and still validates its original functional intent (semantic assertions unchanged).
2. **Given** a test exceeding the defined soft budget (20s) in the future, **When** the suite runs, **Then** a clear report highlights the offending test(s) and provides guidance to reduce scope (e.g., scenario count, episode count) before merging.
3. **Given** an intentionally complex integration test, **When** it risks exceeding the hard timeout (60s), **Then** it is forcibly terminated, marked as failed, and emits actionable messaging about next optimization steps.
4. **Given** the benchmark resume test, **When** it is run twice, **Then** the second execution performs a fast no-op (no duplicate records) and total wall time for both sequential runs remains under the defined limit.

### Edge Cases
- Extremely slow host machine: Tests still enforce an upper hard timeout (60s) preventing indefinite hangs.
- Test depends on large scenario matrix: It must supply a minimized deterministic matrix for performance-focused mode while preserving logical coverage.
- Flaky timing variance in CI: Soft performance guidance (advisory) is not treated as a failure unless a configurable threshold is crossed.
- Future addition of new benchmark tests: Missing performance annotations should be detected during review and flagged by documentation.

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: Each test MUST complete within a hard timeout of 60 seconds (enforced individually).
- **FR-002**: Previously identified slow benchmark tests MUST be refactored so their median local runtime is below 20 seconds.
- **FR-003**: Slow benchmark-style tests MUST use minimal deterministic input datasets/scenario matrices while retaining semantic validation (e.g., reproducibility, resume correctness).
- **FR-004**: The system MUST provide a mechanism to detect and report the top N slowest tests after a run (for maintainers to act on) [NEEDS CLARIFICATION: exact N not specified].
- **FR-005**: Resume-related tests MUST confirm no duplication of artifacts on second execution while remaining within the time budget.
- **FR-006**: Tests introducing performance optimizations MUST retain original assertion intent (no weakening of correctness checks allowed).
- **FR-007**: A documentation update MUST define the per-test performance budget and optimization guidelines for contributors.
- **FR-008**: A clear policy MUST exist for handling tests that repeatedly violate budgets (e.g., mark xfail, skip, or block merge) [NEEDS CLARIFICATION: enforcement mechanism].
- **FR-009**: The suite MUST allow an override environment variable to relax timing assertions for constrained CI runners (e.g., slow virtualized hardware) [NEEDS CLARIFICATION: variable name].
- **FR-010**: The optimization process MUST not remove critical coverage of benchmark orchestration logic (episode generation, resume semantics, configuration parsing).
- **FR-011**: A changelog or dev documentation entry MUST summarize the new performance expectations.

### Key Entities
- **Test Performance Budget**: Conceptual policy defining soft (<20s) and hard (60s) thresholds per test; includes optional environment overrides.
- **Minimal Scenario Matrix**: A reduced single-scenario input used by performance-sensitive tests to bound runtime while preserving logical validation paths.
- **Slow Test Report**: A summary artifact (console output) listing slowest tests with guidance.

## Review & Acceptance Checklist

### Content Quality
- [ ] No implementation details (code-level specifics avoided)
- [ ] Focused on user/maintenance value
- [ ] Written for stakeholders concerned with reliability and velocity
- [ ] Mandatory sections completed

### Requirement Completeness
- [ ] All [NEEDS CLARIFICATION] items resolved (FR-004, FR-008, FR-009)
- [ ] Requirements testable & unambiguous post-clarification
- [ ] Success criteria measurable (per-test runtime, pass/fail on timeout)
- [ ] Scope bounded to test performance (not core engine optimization)
- [ ] Assumptions documented (deterministic mini inputs feasible)

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed (pending clarification resolution)

## Open Clarifications
- What is the target N for slowest test reporting? (Candidate: 10)
- What enforcement action occurs when a test repeatedly exceeds 20s but under 60s? (Advisory vs soft fail?)
- Name of environment variable to relax soft performance assertion? (e.g., TEST_PERF_RELAX=1 or ROBOT_SF_SLOW_TEST_OK=1)

## Success Metrics
- 100% of tests complete <60s (hard) and ≥95% complete <20s (soft) on a reference machine.
- Reduction in total suite runtime versus pre-optimization baseline (baseline to be captured before rollout).
- Zero loss of functional assertion coverage (no removed critical checks).

## Risks & Mitigations
- Risk: Over-tight timing causes flakiness → Mitigation: Use advisory soft threshold plus opt-out env var.
- Risk: Reduced scenarios miss edge conditions → Mitigation: Retain at least one deterministic path per logical branch validated by existing assertions.
- Risk: CI variability inflates durations → Mitigation: Hard timeout generous (60s) and soft check configurable.

## Out of Scope
- Core simulation engine performance tuning.
- Architectural rework of benchmark orchestrator.
- Parallel execution strategy changes.

## Rollout Plan
1. Capture baseline durations (current top 10 slow tests).
2. Optimize identified benchmark tests (resume, reproducibility, full classic variants).
3. Add timeout annotations and minimal input fixtures.
4. Introduce slow test reporting hook.
5. Update documentation and changelog.
6. Final review—remove [NEEDS CLARIFICATION] markers.

## Assumptions
- Deterministic minimal inputs can replicate logical correctness conditions.
- Contributors will adhere to documented performance budget once formalized.

## Dependencies
- Existing benchmark orchestration functions remain stable.
- Pytest available for timeout markers.

## Glossary
- Soft Threshold: Advisory runtime goal per test (<20s).
- Hard Timeout: Maximum enforced runtime before forced failure (60s).
