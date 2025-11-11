# Feature Specification: Document Configuration Hierarchy and Deprecate Legacy Config Classes

**Feature Branch**: `244-document-config-hierarchy`  
**Created**: 2025-01-11  
**Status**: Draft  
**Input**: Issue #244 - Document configuration hierarchy and deprecate legacy config classes

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Configuration Precedence Documentation (Priority: P1)

As a **developer integrating robot_sf**, I want to understand which configuration takes priority (code defaults vs YAML files vs runtime parameters) so I can confidently override settings without unexpected behavior.

**Why this priority**: Critical foundation - without clear precedence rules, developers will make incorrect assumptions leading to bugs and frustration. This is the minimum viable documentation.

**Independent Test**: Can be fully tested by creating `docs/architecture/configuration.md` with precedence rules, then validating that a developer can follow it to successfully override a config parameter.

**Acceptance Scenarios**:

1. **Given** a new developer reads the configuration docs, **When** they need to override a default parameter, **Then** they can determine the correct level (code/YAML/runtime) to apply their override
2. **Given** configuration precedence is documented, **When** a developer sets the same parameter at multiple levels, **Then** they can predict which value will be used
3. **Given** the documentation exists, **When** a developer searches for "configuration" in docs, **Then** they find the architecture/configuration.md page

---

### User Story 2 - Legacy Config Identification (Priority: P2)

As a **maintainer reviewing code**, I want legacy config classes clearly marked as deprecated so I can guide contributors toward the unified config system and plan removal.

**Why this priority**: Prevents new code from using legacy patterns, reduces technical debt accumulation, and enables future cleanup without breaking existing code.

**Independent Test**: Can be tested independently by adding deprecation warnings to `env_config.py` and `sim_config.py`, then running tests to verify warnings appear but functionality remains unchanged.

**Acceptance Scenarios**:

1. **Given** legacy config classes exist, **When** they are instantiated, **Then** a DeprecationWarning is emitted with migration guidance
2. **Given** deprecation warnings are added, **When** tests run, **Then** all tests pass (non-breaking change)
3. **Given** a developer uses a legacy config class, **When** they see the warning, **Then** they know which modern config class to use instead

---

### User Story 3 - Configuration Migration Guide (Priority: P3)

As a **user upgrading robot_sf**, I want a migration guide showing how to convert from legacy configs to unified configs so I can modernize my codebase without breaking changes.

**Why this priority**: Helps users proactively migrate, but not critical since legacy configs remain functional with deprecation warnings.

**Independent Test**: Can be tested by creating migration guide with code examples, then having a user successfully convert a legacy config to unified config following the guide.

**Acceptance Scenarios**:

1. **Given** a migration guide exists, **When** a user has legacy `EnvSettings` or `PedEnvSettings`, **Then** they can find the equivalent unified config example (Note: SimulationSettings is canonical, not legacy)
2. **Given** migration examples are provided, **When** a user follows them, **Then** their code works identically with the new config
3. **Given** the guide is linked from dev_guide.md, **When** users search for configuration docs, **Then** they discover the migration path

---

### User Story 4 - Config Module Structure Documentation (Priority: P4)

As a **contributor planning future work**, I want the current config module structure documented so I understand which modules are canonical vs legacy when adding features.

**Why this priority**: Nice-to-have for maintainers, but lower priority than user-facing documentation.

**Independent Test**: Can be tested by documenting the module structure in configuration.md, then verifying a contributor can identify the correct module for a new config parameter.

**Acceptance Scenarios**:

1. **Given** config modules are documented, **When** adding a new robot parameter, **Then** the contributor knows to use `unified_config.py`
2. **Given** legacy modules are identified, **When** reviewing a PR, **Then** maintainers can quickly spot incorrect config usage
3. **Given** the structure is documented, **When** planning Phase 3 consolidation, **Then** the team has a clear baseline to refactor from

### Edge Cases

- What happens when a parameter is set at all three levels (code default, YAML file, runtime kwarg)? → Runtime kwarg takes precedence, documented clearly
- How does system handle a deprecated config class mixed with unified config in the same codebase? → Both work independently, deprecation warning guides migration
- What if YAML file references a parameter that doesn't exist in the unified config schema? → Validation should catch this (existing behavior)
- What if a user depends on legacy config behavior that differs from unified config? → Migration guide documents behavioral differences if any exist
- How are nested configuration parameters (e.g., fast-pysf config) handled in the precedence hierarchy? → Documented as external config with pass-through behavior

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: Documentation MUST clearly define configuration precedence: Code Defaults < YAML Files < Runtime Parameters
- **FR-002**: Documentation MUST identify all configuration modules and their status (canonical vs legacy)
- **FR-003**: Legacy config classes MUST emit DeprecationWarning when instantiated
- **FR-004**: Deprecation warnings MUST include the recommended replacement class
- **FR-005**: All existing tests MUST pass after adding deprecation warnings (non-breaking)
- **FR-006**: Migration guide MUST provide code examples for each legacy config → unified config conversion
- **FR-007**: `dev_guide.md` MUST reference the new configuration documentation
- **FR-008**: Documentation MUST explain how YAML configs map to unified config classes
- **FR-009**: Documentation MUST clarify the relationship with external configs (fast-pysf)
- **FR-010**: Documentation MUST defer Phase 3 (consolidation) to future work with clear rationale

### Key Entities

- **Configuration Document** (`docs/architecture/configuration.md`): Central reference explaining hierarchy, module structure, precedence rules, and migration paths
- **Legacy Config Classes**: `BaseEnvSettings`, `RobotEnvSettings`, `EnvSettings`, `PedEnvSettings` (in `env_config.py`)
- **Canonical Config Classes**: `BaseSimulationConfig`, `RobotSimulationConfig`, `ImageRobotConfig`, `PedestrianSimulationConfig` (in `unified_config.py`)
- **SimulationSettings** (in `sim_config.py`): Still canonical - used by unified config classes
- **Canonical Config Classes**: `BaseSimulationConfig`, `RobotSimulationConfig`, `ImageRobotConfig`, `PedestrianSimulationConfig` (in `unified_config.py`)
- **YAML Configs**: Scenario and baseline configuration files in `configs/scenarios/` and `configs/baselines/`
- **Migration Guide**: Section in configuration.md or separate doc showing legacy → unified conversions

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: Developer can determine configuration precedence in under 2 minutes by reading `docs/architecture/configuration.md`
- **SC-002**: All tests pass after adding deprecation warnings (no functionality breaks)
- **SC-003**: Deprecation warnings appear in test output for any code using legacy configs
- **SC-004**: User can successfully migrate one legacy config to unified config following migration guide examples
- **SC-005**: Configuration documentation is discoverable via search from `docs/README.md` and `dev_guide.md`
- **SC-006**: Zero new usage of legacy config classes in future PRs (after documentation is merged)
- **SC-007**: Configuration precedence hierarchy matches actual implementation behavior (verified by tests)
