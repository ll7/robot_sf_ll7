# Feature Specification: Architectural decoupling and consistency overhaul

**Feature Branch**: `149-architectural-coupling-and`  
**Created**: 2025-10-29  
**Status**: Draft  
**Input**: User description: "Architectural coupling and design inconsistencies (tight coupling among Simulator/PedSimulator/sensors, inconsistent sensor fusion patterns, mixed abstraction levels, inconsistent error handling, complex configuration hierarchy)."

## User Scenarios & Testing (mandatory)

### User Story 1 - Swap simulation backends without touching env code (Priority: P1)

As a maintainer, I can replace or upgrade the pedestrian simulation backend (e.g., SocialForce variant) via a single factory/config change, without editing environment classes or sensors.

Why this priority: This directly addresses tight coupling and reduces regression risk when evolving simulation internals.

Independent Test: Provide two minimal backends implementing the same Simulator Facade. Switch via config and run an env reset→step smoke without modifying env code.

Acceptance Scenarios:

1. Given an environment created via factory, When the simulator implementation is toggled in config, Then the environment resets and steps successfully with identical public observations/actions shape.
2. Given baseline training scripts, When backend is switched, Then no import paths or call sites outside the factory need edits.

---

### User Story 2 - Add a new sensor without editing fusion or simulator (Priority: P1)

As a developer, I can implement a new sensor by conforming to a small sensor interface and register it in config, without changing SensorFusion or Simulator types.

Why this priority: Decouples sensing from fusion/simulator, reduces integration friction.

Independent Test: Implement a dummy sensor (returns constants) and enable it via config; fusion stack consumes it and env produces a stable observation dict.

Acceptance Scenarios:

1. Given SensorFusion configured with [sensor_a, sensor_b, new_sensor], When env.step runs, Then fusion receives three inputs and outputs the same schema (with new channel if configured) without code changes in fusion or simulator.
2. Given the new sensor disabled in config, When env.reset/step executes, Then outputs and performance remain unchanged.

---

### User Story 3 - Graceful error handling with actionable messages (Priority: P2)

As a user, when a required component is missing (e.g., map assets, model weights), I receive a clear error with remediation steps, and optional soft-degrade paths where appropriate.

Why this priority: Reduces support burden and improves reliability.

Independent Test: Remove a required asset and run a demo; verify the error message includes the missing path and a one-line fix. For optional components, verify warnings and fallbacks activate.

Acceptance Scenarios:

1. Given missing map file, When env.reset is called, Then a single RuntimeError is raised with the exact file path and a short “How to fix” line.
2. Given optional recording enabled but ffmpeg unavailable, When recording starts, Then a WARNING is logged and run continues without crashing.

---

### User Story 4 - Consistent configuration with validation (Priority: P2)

As a developer, I can define env/simulator/sensor/fusion options in a unified config with schema validation and helpful defaults.

Why this priority: Eliminates overlapping configs and lowers onboarding time.

Independent Test: Provide an invalid config (unknown sensor name) and verify the error explains allowed values. Provide a minimal config and verify defaults are applied deterministically.

Acceptance Scenarios:

1. Given a config referencing an unknown backend key, When creating the env via factory, Then validation fails with a message listing valid keys.
2. Given a minimal config, When env is created, Then defaulted values are materialized and observable via a “resolved config” dump.

### Edge Cases

- Switching backends mid-run (after reset) is disallowed; attempts return a clear error detailing permitted lifecycle stages.
- Sensors producing NaNs: fusion detects and replaces with safe defaults while logging source sensor names and counts.
- Conflicting options (e.g., image and non-image fusion stacks enabled simultaneously) are rejected with a conflict message listing mutually exclusive flags.

## Requirements (mandatory)

### Functional Requirements

- **FR-001**: Provide a Simulator Facade with a minimal stable interface (create, reset, step, get_state) consumed by envs via factory.
- **FR-002**: Support pluggable simulator implementations registered by a unique key; selection is driven by unified config.
- **FR-003**: Define a Sensor interface (init, read, close) and a Fusion interface (register sensors, compose outputs) so that adding a sensor requires no changes to fusion or simulator code.
- **FR-004**: Enforce consistent error handling policy: fatal for required resources (with remediation), warning with fallback for optional ones; document the policy and apply across env/sim/sensors/fusion.
- **FR-005**: Consolidate configuration into a unified, schema-validated structure with defaults and conflict detection.
- **FR-006**: Expose a resolved-config inspection endpoint or log for reproducibility (single source of truth per run).
- **FR-007**: Ensure testability: each component (simulator, sensor, fusion) can be instantiated and tested in isolation with lightweight stubs.
- **FR-008**: Preserve public API compatibility for environment factories and observation/action schemas unless explicitly versioned and documented.

### Key Entities

- **Simulator Facade**: A stable contract consumed by envs; abstracts underlying physics engines.
- **Sensor**: Produces typed observation slices with a name and shape; lifecycle managed by env/fusion.
- **Fusion Pipeline**: Composes multiple sensor outputs into a single observation per step; configurable stacking/merging policy.
- **Unified Config**: Hierarchical configuration capturing env, simulator, sensors, fusion; validated against a schema.

## Success Criteria (mandatory)

### Measurable Outcomes

- **SC-001**: Environment code requires 0 source changes when switching simulator backends via config (verified by diff on env modules).
- **SC-002**: Add-a-sensor workflow takes ≤ 30 minutes for a developer familiar with the codebase and requires 0 changes to fusion or simulator modules.
- **SC-003**: 95% of fatal errors include a remediation line (“How to fix”) and the missing path/value; optional-path issues log warnings and continue.
- **SC-004**: Unified config validation rejects conflicting or unknown keys with messages under 200 characters and lists allowed alternatives.
- **SC-005**: Component-level unit tests run independently (simulator, sensors, fusion) and complete in ≤ 60 seconds total on a standard laptop.

### Assumptions

- Existing env factory functions remain the primary entry points; we will not introduce breaking changes to their signatures in this feature.
- Observation and action schemas remain stable for the initial rollout; any schema evolution will be versioned separately.
- The codebase already uses logging and schema validation libraries consistent with repository standards.
# Feature Specification: [FEATURE NAME]

**Feature Branch**: `[###-feature-name]`  
**Created**: [DATE]  
**Status**: Draft  
**Input**: User description: "$ARGUMENTS"

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

### User Story 1 - [Brief Title] (Priority: P1)

[Describe this user journey in plain language]

**Why this priority**: [Explain the value and why it has this priority level]

**Independent Test**: [Describe how this can be tested independently - e.g., "Can be fully tested by [specific action] and delivers [specific value]"]

**Acceptance Scenarios**:

1. **Given** [initial state], **When** [action], **Then** [expected outcome]
2. **Given** [initial state], **When** [action], **Then** [expected outcome]

---

### User Story 2 - [Brief Title] (Priority: P2)

[Describe this user journey in plain language]

**Why this priority**: [Explain the value and why it has this priority level]

**Independent Test**: [Describe how this can be tested independently]

**Acceptance Scenarios**:

1. **Given** [initial state], **When** [action], **Then** [expected outcome]

---

### User Story 3 - [Brief Title] (Priority: P3)

[Describe this user journey in plain language]

**Why this priority**: [Explain the value and why it has this priority level]

**Independent Test**: [Describe how this can be tested independently]

**Acceptance Scenarios**:

1. **Given** [initial state], **When** [action], **Then** [expected outcome]

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right edge cases.
-->

- What happens when [boundary condition]?
- How does system handle [error scenario]?

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: System MUST [specific capability, e.g., "allow users to create accounts"]
- **FR-002**: System MUST [specific capability, e.g., "validate email addresses"]  
- **FR-003**: Users MUST be able to [key interaction, e.g., "reset their password"]
- **FR-004**: System MUST [data requirement, e.g., "persist user preferences"]
- **FR-005**: System MUST [behavior, e.g., "log all security events"]

*Example of marking unclear requirements:*

- **FR-006**: System MUST authenticate users via [NEEDS CLARIFICATION: auth method not specified - email/password, SSO, OAuth?]
- **FR-007**: System MUST retain user data for [NEEDS CLARIFICATION: retention period not specified]

### Key Entities *(include if feature involves data)*

- **[Entity 1]**: [What it represents, key attributes without implementation]
- **[Entity 2]**: [What it represents, relationships to other entities]

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: [Measurable metric, e.g., "Users can complete account creation in under 2 minutes"]
- **SC-002**: [Measurable metric, e.g., "System handles 1000 concurrent users without degradation"]
- **SC-003**: [User satisfaction metric, e.g., "90% of users successfully complete primary task on first attempt"]
- **SC-004**: [Business metric, e.g., "Reduce support tickets related to [X] by 50%"]
