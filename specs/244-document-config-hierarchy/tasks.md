# Tasks: Document Configuration Hierarchy and Deprecate Legacy Config Classes

**Feature**: Issue #244  
**Branch**: `244-document-config-hierarchy`
**Status**: Ready for Implementation

## Task Overview

This feature implements configuration hierarchy documentation and legacy config deprecation through 4 independent user stories, organized by priority. Each story is independently testable and can be implemented incrementally.

**Total Tasks**: 47  
**Estimated Effort**: 4-6 hours  
**MVP Scope**: User Story 1 only (Configuration Precedence Documentation)

---

## Phase 1: Setup

**Goal**: Prepare project structure and verify baseline

### Tasks

- [X] T001 Verify project structure matches plan.md specifications
- [X] T002 Create docs/architecture/ directory if not exists
- [X] T003 Verify tests/test_gym_env/ directory exists for new test file
- [X] T004 Run baseline test suite to establish passing state: `uv run pytest tests -v`

**Completion Criteria**: All setup tasks complete, baseline tests passing (pre-change state captured)

---

## Phase 2: User Story 1 - Configuration Precedence Documentation (P1)

**Story Goal**: Enable developers to understand configuration precedence rules (Code < YAML < Runtime)

**Independent Test Criteria**:
- `docs/architecture/configuration.md` exists
- Document includes precedence hierarchy section with examples
- Developer can determine correct override level by reading the doc

### Tasks

- [X] T005 [US1] Create docs/architecture/configuration.md with document structure outline
- [X] T006 [US1] Write "Overview" section explaining configuration system purpose and scope
- [X] T007 [US1] Write "Precedence Hierarchy" section with Code < YAML < Runtime explanation and code examples
- [X] T008 [US1] Add "Configuration Sources" section documenting code defaults, YAML files, runtime parameters
- [X] T009 [US1] Add "Best Practices" section with guidance on when to use each configuration level
- [X] T010 [US1] Update docs/README.md to link to architecture/configuration.md in appropriate section
- [X] T011 [US1] Update docs/dev_guide.md to reference configuration documentation in "Configuration hierarchy" section

**Acceptance Validation**:
```bash
# Manual verification
cat docs/architecture/configuration.md | grep -i "precedence"
grep "configuration.md" docs/README.md
grep "configuration" docs/dev_guide.md
```

**Story Complete When**: Developer can read configuration.md and determine which override level (code/YAML/runtime) to use for their scenario

---

## Phase 3: User Story 2 - Legacy Config Identification (P2)

**Story Goal**: Mark legacy config classes as deprecated to guide contributors toward unified config system

**Independent Test Criteria**:

- Deprecation warnings emit when legacy classes are instantiated- [ ] **Task 2.2**: Add deprecation warning to `env_config.py::PedEnvSettings`

- All existing tests pass (backward compatibility maintained)  - **Estimate**: 30 minutes

- Warning messages include replacement class names  - **Dependencies**: Task 1.3

  - **Acceptance**: DeprecationWarning emitted with message pointing to `PedestrianSimulationConfig`

### Tasks

- [ ] **Task 2.3**: Add deprecation warning to `sim_config.py::SimulationSettings`

- [ ] T012 [P] [US2] Create tests/test_gym_env/test_config_deprecation.py with test structure  - **Estimate**: 30 minutes

- [ ] T013 [P] [US2] Add test_base_env_settings_deprecated() to verify BaseEnvSettings emits DeprecationWarning  - **Dependencies**: Task 1.3

- [ ] T014 [P] [US2] Add test_robot_env_settings_deprecated() to verify RobotEnvSettings emits DeprecationWarning  - **Acceptance**: DeprecationWarning emitted with message pointing to appropriate unified config

- [ ] T015 [P] [US2] Add test_env_settings_deprecated() to verify EnvSettings emits DeprecationWarning (if distinct from RobotEnvSettings)

- [ ] T016 [P] [US2] Add test_ped_env_settings_deprecated() to verify PedEnvSettings emits DeprecationWarning- [ ] **Task 2.4**: Verify all tests still pass with deprecation warnings

- [ ] T017 [US2] Add deprecation warning to BaseEnvSettings.__post_init__() in robot_sf/gym_env/env_config.py  - **Estimate**: 1 hour

- [ ] T018 [US2] Add deprecation warning to RobotEnvSettings.__post_init__() in robot_sf/gym_env/env_config.py  - **Dependencies**: Tasks 2.1, 2.2, 2.3

- [ ] T019 [US2] Add deprecation warning to EnvSettings.__post_init__() in robot_sf/gym_env/env_config.py (if distinct class)  - **Acceptance**: `uv run pytest tests` passes; warnings appear but don't fail tests

- [ ] T020 [US2] Add deprecation warning to PedEnvSettings.__post_init__() in robot_sf/gym_env/env_config.py  - **Actions**:

- [ ] T021 [US2] Run deprecation tests to verify warnings emit correctly: `uv run pytest tests/test_gym_env/test_config_deprecation.py -v`    - Run full test suite

- [ ] T022 [US2] Run full test suite to verify backward compatibility: `uv run pytest tests -v`    - Verify warnings appear in test output

    - Document any test failures (should be none)

**Deprecation Warning Pattern**:    - Update tests if needed to suppress warnings where appropriate

```python

import warnings### Phase 3: Migration Guide (P3 - Nice to Have)



def __post_init__(self):- [ ] **Task 3.1**: Create migration guide section in configuration.md

    warnings.warn(  - **Estimate**: 2 hours

        "{LegacyClass} is deprecated and will be removed in a future version. "  - **Dependencies**: Task 2.4

        "Use {CanonicalClass} from robot_sf.gym_env.unified_config instead.",  - **Acceptance**: Section with code examples for each legacy → unified conversion

        DeprecationWarning,  - **Content includes**:

        stacklevel=2    - `SimulationSettings` → `RobotSimulationConfig` example

    )    - `EnvSettings` → `RobotSimulationConfig` example

    # ... existing validation code ...    - `PedEnvSettings` → `PedestrianSimulationConfig` example

```    - Parameter mapping table showing old → new names if any differ

    - Common gotchas and differences in behavior

**Mapping**:

- BaseEnvSettings → BaseSimulationConfig- [ ] **Task 3.2**: Add YAML config examples showing unified config usage

- RobotEnvSettings → RobotSimulationConfig  - **Estimate**: 1 hour

- EnvSettings → RobotSimulationConfig  - **Dependencies**: Task 3.1

- PedEnvSettings → PedestrianSimulationConfig  - **Acceptance**: Examples show how YAML parameters map to unified config classes

  - **Content includes**:

**Acceptance Validation**:    - Example YAML snippet

```bash    - Corresponding unified config instantiation

# Verify warnings appear    - Explanation of how YAML overrides work

uv run pytest tests/test_gym_env/test_config_deprecation.py -v

### Phase 4: Validation and Documentation (P1 - Must Have)

# Verify all tests still pass

uv run pytest tests -v | grep -E "(PASSED|FAILED|ERROR)"- [ ] **Task 4.1**: Add tests validating configuration precedence

```  - **Estimate**: 2 hours

  - **Dependencies**: Task 1.2

**Story Complete When**: All legacy config classes emit deprecation warnings, all tests pass, developers see clear guidance to migrate  - **Acceptance**: Tests verify Code < YAML < Runtime precedence behavior

  - **Test cases**:

---    - Set parameter at code level, verify default used

    - Set parameter in YAML, verify YAML overrides code default

## Phase 4: User Story 3 - Configuration Migration Guide (P3)    - Set parameter at runtime, verify runtime overrides YAML

    - Set at all three levels, verify runtime wins

**Story Goal**: Provide migration guide for users to convert legacy configs to unified configs

- [ ] **Task 4.2**: Run quality gates (Ruff, type check, tests)

**Independent Test Criteria**:  - **Estimate**: 30 minutes

- Migration guide section exists in configuration.md  - **Dependencies**: All above tasks

- Guide includes code examples for each legacy → unified conversion  - **Acceptance**: All quality gates pass

- User can successfully convert a legacy config following the examples  - **Commands**:

    - `uv run ruff check . && uv run ruff format .`

### Tasks    - `uvx ty check . --exit-zero`

    - `uv run pytest tests`

- [ ] T023 [US3] Add "Migration Guide" section to docs/architecture/configuration.md

- [ ] T024 [US3] Document EnvSettings → RobotSimulationConfig migration with before/after code example- [ ] **Task 4.3**: Update CHANGELOG.md

- [ ] T025 [US3] Document PedEnvSettings → PedestrianSimulationConfig migration with before/after code example  - **Estimate**: 15 minutes

- [ ] T026 [US3] Document RobotEnvSettings → RobotSimulationConfig migration with before/after code example  - **Dependencies**: All above tasks

- [ ] T027 [US3] Add migration examples for common scenarios (basic robot, pedestrian, custom settings)  - **Acceptance**: CHANGELOG documents deprecations and new configuration docs

- [ ] T028 [US3] Document field mapping table showing legacy field → canonical field equivalence  - **Entry format**:

- [ ] T029 [US3] Add "Behavioral Differences" subsection noting any runtime differences (if none, state explicitly)    ```markdown

    ### Documentation

**Migration Example Format**:    - Added `docs/architecture/configuration.md` explaining configuration hierarchy

```markdown    - Documented precedence: Code Defaults < YAML Files < Runtime Parameters

### EnvSettings → RobotSimulationConfig    

    ### Deprecated

**Legacy (deprecated)**:    - `env_config.EnvSettings` (use `unified_config.RobotSimulationConfig`)

```python    - `env_config.PedEnvSettings` (use `unified_config.PedestrianSimulationConfig`)

from robot_sf.gym_env.env_config import EnvSettings    - `sim_config.SimulationSettings` (use `unified_config.RobotSimulationConfig`)

config = EnvSettings()    ```

```

## Deferred to Phase 3 (Future Work)

**Unified (recommended)**:

```pythonThe following is explicitly **NOT** part of this issue (deferred to v3.0):

from robot_sf.gym_env.unified_config import RobotSimulationConfig

config = RobotSimulationConfig()- ❌ Consolidating all config modules into single `robot_sf/config/` package

```- ❌ Removing legacy config classes (breaking change)

- ❌ Refactoring existing code to use unified configs exclusively

**Note**: Field names and structure are identical; only import path changes.- ❌ Schema validation for YAML configs

```

**Rationale**: Phase 3 requires breaking changes suitable for major version bump. This issue focuses on non-breaking documentation and deprecation warnings only.

**Acceptance Validation**:

```bash## Effort Estimate

# Verify migration guide exists

grep -A 5 "Migration Guide" docs/architecture/configuration.md| Phase | Tasks | Estimated Time |

|-------|-------|----------------|

# Verify code examples present| Phase 1: Documentation | 1.1-1.5 | ~4.5 hours |

grep "from robot_sf.gym_env.env_config import" docs/architecture/configuration.md| Phase 2: Deprecation | 2.1-2.4 | ~2.5 hours |

grep "from robot_sf.gym_env.unified_config import" docs/architecture/configuration.md| Phase 3: Migration Guide | 3.1-3.2 | ~3 hours |

```| Phase 4: Validation | 4.1-4.3 | ~2.75 hours |

| **Total** | **14 tasks** | **~12.75 hours** |

**Story Complete When**: User can locate migration guide, follow code examples, and successfully convert legacy config to unified config

## Success Criteria Checklist

---

- [ ] SC-001: Developer can determine precedence in < 2 minutes from docs

## Phase 5: User Story 4 - Config Module Structure Documentation (P4)- [ ] SC-002: All tests pass after deprecation warnings

- [ ] SC-003: Deprecation warnings appear in test output

**Story Goal**: Document config module structure so contributors know which modules are canonical vs legacy- [ ] SC-004: User can migrate legacy → unified following guide

- [ ] SC-005: Configuration docs discoverable from README and dev_guide

**Independent Test Criteria**:- [ ] SC-006: Zero new legacy config usage in future PRs

- Module structure section exists in configuration.md- [ ] SC-007: Precedence hierarchy matches implementation (verified by tests)

- Canonical and legacy modules clearly identified

- Contributor can determine correct module for new config parameter## Notes



### Tasks- All changes are non-breaking (MINOR or PATCH version bump)

- Deprecation warnings use `stacklevel=2` to show caller location

- [ ] T030 [US4] Add "Configuration Modules" section to docs/architecture/configuration.md- Tests may need `warnings.filterwarnings()` to suppress deprecations in test fixtures

- [ ] T031 [US4] Document canonical module: robot_sf/gym_env/unified_config.py with class list and purposes- Migration guide examples should be copy-paste ready

- [ ] T032 [US4] Document legacy modules: robot_sf/gym_env/env_config.py with deprecation status
- [ ] T033 [US4] Add "Unified Config Classes" subsection documenting each canonical class hierarchy
- [ ] T034 [P] [US4] Document BaseSimulationConfig fields and purpose
- [ ] T035 [P] [US4] Document RobotSimulationConfig fields and purpose (extends BaseSimulationConfig)
- [ ] T036 [P] [US4] Document ImageRobotConfig fields and purpose (extends RobotSimulationConfig)
- [ ] T037 [P] [US4] Document PedestrianSimulationConfig fields and purpose (extends RobotSimulationConfig)
- [ ] T038 [US4] Add "YAML Configuration" section explaining how configs/scenarios/ and configs/baselines/ map to unified classes
- [ ] T039 [US4] Add "External Configuration" section documenting fast-pysf config and other external configs
- [ ] T040 [US4] Add "Future Work" section deferring Phase 3 consolidation with clear rationale

**Acceptance Validation**:
```bash
# Verify module documentation
grep "unified_config.py" docs/architecture/configuration.md
grep "env_config.py" docs/architecture/configuration.md
grep "BaseSimulationConfig" docs/architecture/configuration.md
grep "RobotSimulationConfig" docs/architecture/configuration.md
```

**Story Complete When**: Contributor can identify canonical module for new config parameter and understand module status (canonical vs legacy)

---

## Phase 6: Polish & Validation

**Goal**: Finalize implementation and verify all quality gates pass

### Tasks

- [ ] T041 Run quality gate: Format and lint with `uv run ruff check --fix . && uv run ruff format .`
- [ ] T042 Run quality gate: Type check with `uvx ty check . --exit-zero`
- [ ] T043 Run quality gate: Full test suite with `uv run pytest tests -v`
- [ ] T044 Verify deprecation warnings appear in test output (expected behavior)
- [ ] T045 Manual verification: Read docs/architecture/configuration.md for completeness and clarity
- [ ] T046 Manual verification: Verify docs/README.md link to configuration.md works
- [ ] T047 Manual verification: Verify docs/dev_guide.md reference to configuration docs is clear

**Completion Criteria**: All quality gates pass, documentation is complete and discoverable, deprecation warnings working

---

## Dependencies & Execution Strategy

### User Story Dependencies

```
Setup (Phase 1) → Foundational prerequisite for all stories
    ↓
US1 (P1) → Independent, no dependencies (MVP)
    ↓ (optional link for context)
US2 (P2) → Independent (references US1 docs but can stand alone)
    ↓ (optional link for migration context)
US3 (P3) → Depends on US1 (migration guide references precedence docs)
    ↓ (optional link for context)
US4 (P4) → Independent (module docs enhance US1/US3 but not required)
    ↓
Polish → Requires all stories complete
```

**Key Insight**: US1, US2, and US4 are truly independent. US3 has soft dependency on US1 for context but can be implemented separately.

### Parallel Execution Opportunities

**Within User Story 2 (P2)**:
- Tasks T012-T016 (test creation) can run in parallel - all create tests in same file but different functions
- After tests exist, tasks T017-T020 (add warnings) can run in parallel - different classes in same file

**Within User Story 4 (P4)**:
- Tasks T034-T037 (document config classes) can run in parallel - different sections
- Tasks T038-T040 (YAML, external, future work) can run in parallel - different sections

**Example Parallel Batch for US2**:
```bash
# Parallel: Create all deprecation tests simultaneously
run_task T013 &  # BaseEnvSettings test
run_task T014 &  # RobotEnvSettings test
run_task T015 &  # EnvSettings test
run_task T016 &  # PedEnvSettings test
wait

# Sequential: Add warnings to code
run_task T017  # BaseEnvSettings warning
run_task T018  # RobotEnvSettings warning
run_task T019  # EnvSettings warning
run_task T020  # PedEnvSettings warning

# Sequential: Validate
run_task T021  # Run deprecation tests
run_task T022  # Run full suite
```

---

## Implementation Strategy

### Minimum Viable Product (MVP)

**Scope**: User Story 1 only (Tasks T001-T011)

**Deliverable**: Configuration precedence documentation that enables developers to understand override behavior

**Value**: Provides immediate clarity on configuration system without code changes

**Time**: ~2 hours

**Validation**:
```bash
# MVP complete when these succeed
cat docs/architecture/configuration.md | grep "Precedence Hierarchy"
grep "configuration.md" docs/README.md
grep "configuration" docs/dev_guide.md
```

### Incremental Delivery Plan

**Iteration 1 (MVP)**: US1 - Documentation foundation
- Deliverable: Precedence docs + index links
- Time: 2 hours
- Risk: Low

**Iteration 2**: US2 - Deprecation warnings
- Deliverable: Legacy class warnings + tests
- Time: 1.5 hours
- Risk: Low (non-breaking, tested)

**Iteration 3**: US3 - Migration guide
- Deliverable: Migration examples and code snippets
- Time: 1 hour
- Risk: Low (documentation only)

**Iteration 4**: US4 - Module structure docs
- Deliverable: Module and class documentation
- Time: 1.5 hours
- Risk: Low (documentation only)

**Total**: 4-6 hours across 4 iterations

---

## Success Metrics

### Per-Story Validation

**US1 Success**:
- [ ] docs/architecture/configuration.md exists with precedence section
- [ ] docs/README.md links to configuration.md
- [ ] docs/dev_guide.md references configuration docs
- [ ] Developer can determine override level in < 2 minutes

**US2 Success**:
- [ ] All 4 legacy classes emit DeprecationWarning on instantiation
- [ ] Warning messages include canonical replacement class names
- [ ] All existing tests pass (backward compatibility maintained)
- [ ] Deprecation tests pass: `uv run pytest tests/test_gym_env/test_config_deprecation.py -v`

**US3 Success**:
- [ ] Migration guide section exists with code examples
- [ ] Each legacy → unified conversion documented
- [ ] Field mapping table present
- [ ] User can successfully convert legacy config following guide

**US4 Success**:
- [ ] Module structure documented (canonical vs legacy)
- [ ] All 4 unified config classes documented
- [ ] YAML configuration section present
- [ ] Contributor can identify correct module for new parameter

### Overall Feature Success

**Functional Requirements Met**:
- [x] FR-001: Configuration precedence clearly defined (US1)
- [x] FR-002: Module status identified (US4)
- [x] FR-003: Legacy classes emit DeprecationWarning (US2)
- [x] FR-004: Warnings include replacement class (US2)
- [x] FR-005: All tests pass (US2)
- [x] FR-006: Migration guide with examples (US3)
- [x] FR-007: dev_guide.md references docs (US1)
- [x] FR-008: YAML mapping explained (US4)
- [x] FR-009: External config documented (US4)
- [x] FR-010: Phase 3 deferred (US4)

**Quality Gates**:
- [ ] Ruff clean: `uv run ruff check .`
- [ ] Type check: `uvx ty check . --exit-zero`
- [ ] All tests pass: `uv run pytest tests -v`
- [ ] Documentation discoverable from central index

---

## Notes

**Testing Philosophy**: This feature uses manual verification for documentation quality (US1, US3, US4) and automated tests for code behavior (US2). No TDD approach requested in spec; tests for US2 validate deprecation behavior only.

**Backward Compatibility**: All changes are non-breaking. Legacy configs remain functional with warnings. Migration is voluntary and can happen at user's pace.

**Future Work**: Phase 3 (legacy class removal) is explicitly deferred. This feature sets the foundation for eventual cleanup but does not perform it.

**Performance**: Deprecation warnings have negligible overhead (~microseconds per instantiation). Documentation has zero runtime impact.
