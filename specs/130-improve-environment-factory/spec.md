# Feature Specification: Improve Environment Factory Ergonomics

**Feature Branch**: `130-improve-environment-factory`  
**Created**: 2025-09-22  
**Status**: Updated (post-implementation alignment)  
**Input**: User description: "Improve environment_factory API ergonomics: user-friendly explicit parameters, structured option objects, better docstrings, discoverability helpers, backward-compatible migration plan"

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a developer integrating Robot SF environments, I want clear, discoverable factory functions (`make_robot_env`, `make_image_robot_env`, `make_pedestrian_env`, `make_multi_robot_env`) with explicit, well-documented parameters so I can configure simulation behavior confidently without digging through internal code or relying on `**kwargs` guesswork.

### Acceptance Scenarios
1. **Given** a developer types `make_robot_env(` in an IDE, **When** signature help appears, **Then** all primary configuration parameters (config object, debug flag, recording options, seeding hooks, optional callbacks) are visible with docstring summaries.
2. **Given** a developer wants to enable video recording, **When** they call the factory with `record_video=True` and optionally `video_path`, **Then** the environment attaches a rendering component and recording works without requiring a separate manual setup.
3. **Given** existing code that used the prior generic `**kwargs` version, **When** the library is updated, **Then** it continues to work OR a guided deprecation warning points to the new explicit parameter names.
4. **Given** a developer passes an invalid combination (e.g., `record_video=True` but `debug=False`), **Then** a clear warning or validation feedback is produced before (or immediately after) environment creation.
5. **Given** a developer wants to adjust rarely used advanced options, **When** they inspect a structured options object (e.g., `RenderOptions`, `RecordingOptions`), **Then** they can discover advanced fields without cluttering the main factory signature.
6. **Given** a developer reads the generated Sphinx/IDE docs, **Then** each parameter includes purpose, defaults, and behavior (no empty or ambiguous descriptions).
7. **Given** a user migrates from old usage, **When** they run their code, **Then** either no change is required or a single mechanical rename is sufficient (documented in a migration section) with no silent behavior change.
8. **Given** a developer needs quick examples, **When** they open `docs/ENVIRONMENT.md` or new examples section, **Then** minimal, typical, and advanced usage patterns are shown.

### Edge Cases
- What happens when both `debug=True` and `headless` environment variables are set? → Should still create offscreen rendering but warn if contradictory.
-- What if `record_video=True` and no `video_path` provided? → Frames buffered; logging clarifies missing explicit path (documented).
-- What if a user disables the cap via environment variable and captures >100k frames? → Out-of-scope here; note memory implication; unchanged.
-- How are unknown legacy kwargs handled? → Raise in strict mode; warn + map (when possible) in permissive mode toggled by `ROBOT_SF_FACTORY_LEGACY=1` / `ROBOT_SF_FACTORY_STRICT=1`.
-- Should factories validate config object types strictly (e.g., raising TypeError) or coerce? → Strict type expectation; rely on Python exceptions for incorrect types (no implicit coercion).
-- Multi-robot or pedestrian-specific options: Keep unified option dataclasses for now (avoid premature specialization; revisit if >2 ped-only flags emerge).

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: Factories MUST expose explicit, positional-or-keyword parameters (no blanket `**kwargs`) for primary concerns: `config`, `debug`, `seed`, `record_video`, `video_path`, `reward_func`. (`max_episode_steps` deferred – tracked but not implemented this feature cycle.)
- **FR-002**: Each factory MUST include a comprehensive docstring with a one-line summary, param section, return description, and notes on side effects (e.g., enabling rendering subsystem).
- **FR-003**: System MUST provide structured auxiliary dataclasses or TypedDicts (e.g., `RenderOptions`, `RecordingOptions`) for secondary/advanced toggles to avoid bloating signatures beyond ~8 parameters.
- **FR-004**: Factories MUST validate incompatible inputs (e.g., `record_video=True` + `debug=False`) and emit a warning or raise depending on severity (policy: warning with remediation guidance).
- **FR-005**: A deprecation layer MUST capture legacy `**kwargs` usage, mapping known prior keys to new explicit parameters with a warning referencing migration docs.
- **FR-006**: Unknown legacy kwargs MUST trigger a clear error listing accepted parameters and linking to migration docs unless permissive mode is enabled (`ROBOT_SF_FACTORY_LEGACY=1`). Strict enforcement when `ROBOT_SF_FACTORY_STRICT=1`.
- **FR-007**: IDE auto-completion MUST show parameter names deterministically (verified by a signature test similar to `test_environment_factory_signatures`).
- **FR-008**: All factories MUST accept an optional `seed` that, if provided, seeds environment RNG deterministically (document current seeding model and limitations).
- **FR-009**: Recording-related parameters MUST integrate with rendering so that enabling recording always results in valid frame capture conditions (internally auto-enabling required subsystems where feasible).
- **FR-010**: Docstrings MUST describe performance implications for enabling image observations or video recording.
- **FR-011**: Migration guide MUST be added under `docs/dev/issues/130-improve-environment-factory/` with before→after examples.
- **FR-012**: Example snippets MUST be added or updated in `examples/` demonstrating: basic env, image env, recording, pedestrian env, multi-robot (if supported).
- **FR-013**: Tests MUST cover: signature stability, deprecation warnings on legacy kwargs, validation of incompatible combinations, and successful creation across all factory variants.
- **FR-014**: Factories MUST remain pure (no global mutable side-effects) apart from logging and necessary subsystem init.
- **FR-015**: Logging MUST use Loguru and follow Principle XII (no stray prints) with INFO-level creation message and WARNING-level diagnostics.
- **FR-016**: Type hints MUST be complete (no untyped public function parameters) and pass existing type-check stage (no new errors introduced).
- **FR-017**: Performance baseline MUST not degrade environment creation time by more than +5% vs current (enforced via test with threshold constant = 1.05).
- **FR-018**: All added option dataclasses MUST implement `__repr__` or rely on dataclass default for readable debug output.
- **FR-019**: Documentation MUST include a decision note on rejected alternatives (e.g., builder pattern, nested factory registry) for posterity (captured in migration guide & research notes).
- **FR-020**: A quick reference table MUST list parameters, defaults, and relevant notes (will live in docs, not necessarily docstring duplication).
- **FR-021**: Provide a compatibility shim function (e.g., `make_robot_env_legacy`) if needed for transitional code paths (optional if deprecation mapping sufficiently covers).

All previous ambiguities resolved; this spec reflects final decisions for Feature 130. Coverage checklist (T034) confirms all FRs implemented or explicitly deferred.

### Key Entities
- **EnvironmentFactory Interface**: Conceptual contract grouping four public factories; defines input parameter taxonomy and validation rules.
- **RenderOptions**: Advanced visualization and performance toggles (e.g., `ped_velocity_scale`, `max_fps_override`, maybe future overlay flags).
- **RecordingOptions**: Fields controlling frame cap override, codec hints, `max_frames` (ties to recent addition), and `video_path` default policy.
- **DeprecationMap**: Internal mapping from legacy kw names to new parameter names (enables targeted warnings).

## Review & Acceptance Checklist

### Content Quality
- [ ] No implementation details (languages, frameworks, APIs) beyond necessity of docstrings
- [ ] Focused on user value and discoverability
- [ ] Written for cross-functional stakeholders
- [ ] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [ ] Requirements are testable and unambiguous
- [ ] Success criteria measurable (signature tests, warning tests, perf delta)
- [ ] Scope bounded (only factory ergonomics; not refactoring core env classes)
- [ ] Dependencies & assumptions identified (Loguru, existing config objects, test suite)

## Execution Status
- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities resolved
- [x] User scenarios defined
- [x] Requirements generated (updated post-decisions)
- [x] Entities identified
- [x] Review checklist passed

