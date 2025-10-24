# Feature Specification: Reduce Runtime of Reproducibility Integration Test

**Feature Branch**: `123-reduce-runtime-of`  
**Created**: 2025-09-20  
**Status**: Draft  
**Input**: User description: "Reduce runtime of test test_integration_reproducibility.py::test_reproducibility_same_seed which is currently too slow by optimizing setup, minimizing scenario size, and marking as slow if needed. Provide specification for improving reproducibility test performance without losing validation strength."

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a developer running the test suite locally or in CI, I want the reproducibility integration test (`test_integration_reproducibility.py::test_reproducibility_same_seed`) to complete quickly (target < 2s wall clock on a typical dev machine) while still guaranteeing that identical configuration + seed produce identical ordered episode IDs and associated deterministic hashes, so that iterative development isn’t slowed and CI resource usage remains efficient.

### Acceptance Scenarios
1. **Given** the test suite is executed on a clean workspace, **When** the reproducibility test runs, **Then** it finishes under the target runtime threshold (e.g., < 2s locally, < 4s CI) and asserts identical episode IDs across two runs with the same seed.
2. **Given** a deliberate mutation of the scenario matrix or seed, **When** the modified run is compared, **Then** the test detects a difference and fails (negative control scenario implemented internally as a helper, not a separate slow test).
3. **Given** parallel worker variability, **When** the test executes with workers=1 vs workers>1 (if parameterized internally), **Then** ordering determinism (episode_ids sequence) still holds or the test skips the multi-worker branch with a documented rationale.

### Edge Cases
- Scenario matrix with only a single scenario and minimal horizon still produces stable deterministic hash and IDs.
- Zero episodes (misconfiguration) should fail fast with a clear assertion message.
- Environment seed collisions across scenarios should not occur in reduced configuration.
- Test must remain stable even if other tests have produced artifact directories (should isolate its temp output path).

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: Test MUST execute in < 2s on a standard developer machine (informational goal; hard upper CI allowance < 4s) while validating reproducibility.
- **FR-002**: Test MUST verify that two consecutive benchmark planning/execution runs with identical config + seed produce identical ordered `episode_id` sequences.
- **FR-003**: Test MUST isolate output (temp dir) and not depend on or mutate global benchmark artifacts.
- **FR-004**: Test MUST minimize planned episodes (e.g., 2–4 total) while still exercising scenario hashing and seed expansion logic.
- **FR-005**: Test MUST assert equality of scenario matrix hash (if exposed) or manifest `scenario_matrix_hash` across repeats.
- **FR-006**: Test MUST fail with a clear message if any episode ordering difference is detected.
- **FR-007**: Test MUST document (inline comment) rationale for chosen minimal parameters.
- **FR-008**: Test SHOULD skip or parameter-restrict multi-worker execution if it would introduce nondeterministic ordering; if skipping, MUST note reason.
- **FR-009**: Test MUST avoid network, GPU, or heavy video generation paths (ensure smoke/video skip flags set).
- **FR-010**: Test SHOULD raise an explicit assertion if runtime exceeds target threshold (soft performance guard) while allowing CI variance buffer.
- **FR-011**: Test MUST not reduce statistical or logical coverage of reproducibility semantics (IDs + hash + count); metrics content comparison beyond IDs/hash is optional and excluded to keep runtime low.
- **FR-012**: Test SHOULD provide helper function to run minimal benchmark twice to avoid duplication.

### Key Entities
- **Episode ID Sequence**: Ordered list returned/persisted that encodes scenario seed planning; central reproducibility artifact.
- **Scenario Matrix Hash**: Deterministic hash derived from YAML/planning inputs used to confirm configuration identity.
- **Manifest**: Contains `git_hash`, `scenario_matrix_hash`, and counts; only subset needed for assertions to keep parsing minimal.

## Review & Acceptance Checklist

### Content Quality
- [ ] No implementation details (languages, frameworks, APIs)
- [ ] Focused on user value and business needs
- [ ] Written for non-technical stakeholders
- [ ] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [ ] Requirements are testable and unambiguous  
- [ ] Success criteria are measurable
- [ ] Scope is clearly bounded
- [ ] Dependencies and assumptions identified

## Execution Status

- [ ] User description parsed
- [ ] Key concepts extracted
- [ ] Ambiguities marked
- [ ] User scenarios defined
- [ ] Requirements generated
- [ ] Entities identified
- [ ] Review checklist passed

