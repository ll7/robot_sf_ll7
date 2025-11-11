# Implementation Plan: Document Configuration Hierarchy and Deprecate Legacy Config Classes

**Branch**: `244-document-config-hierarchy` | **Date**: 2025-01-11 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/244-document-config-hierarchy/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This feature addresses the need to document the configuration hierarchy and deprecate legacy config classes in the robot_sf project. The primary requirement is to create comprehensive documentation explaining configuration precedence (Code Defaults < YAML Files < Runtime Parameters), identify canonical vs legacy config modules, add deprecation warnings to legacy classes, and provide migration guidance. The technical approach is documentation-first with minimal code changes (adding deprecation warnings) to maintain backward compatibility while guiding users toward the unified configuration system.

## Technical Context

**Language/Version**: Python 3.11+ (project minimum per pyproject.toml)  
**Primary Dependencies**: 
- Existing: dataclasses (stdlib), warnings (stdlib)
- Documentation: Markdown (existing docs structure)
- Testing: pytest (existing test framework)

**Storage**: Filesystem (documentation files in `docs/architecture/`, deprecation warnings in existing `.py` modules)  
**Testing**: pytest for deprecation warning validation  
**Target Platform**: Documentation accessible via filesystem/GitHub; Python code runs on Linux/macOS  
**Project Type**: Documentation + lightweight code changes (deprecation warnings)  
**Performance Goals**: N/A (documentation-only changes; deprecation warnings negligible overhead)  
**Constraints**: 
- Must maintain backward compatibility (non-breaking changes only)
- All existing tests must pass after adding deprecation warnings
- Documentation must be discoverable from existing `docs/README.md` and `dev_guide.md`

**Scale/Scope**: 
- 1 new documentation file (`docs/architecture/configuration.md`)
- 4 legacy config classes to deprecate (`BaseEnvSettings`, `RobotEnvSettings`, `EnvSettings`, `PedEnvSettings` - all in `env_config.py`)
- 4 canonical config classes documented (`BaseSimulationConfig`, `RobotSimulationConfig`, `ImageRobotConfig`, `PedestrianSimulationConfig` in `unified_config.py`)
- **Note**: `SimulationSettings` (in `sim_config.py`) is canonical and used by the unified config system
- Migration guide covering ~3-5 common conversion scenarios

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Relevant Constitution Principles

**✅ Principle IV - Unified Configuration & Deterministic Seeds**: This feature directly supports this principle by documenting the unified configuration layer and clarifying the deprecation path from ad-hoc kwargs to typed config objects.

**✅ Principle VII - Backward Compatibility & Evolution Gates**: Adding deprecation warnings with migration guidance follows the required deprecation path. No breaking changes - legacy configs remain functional with warnings.

**✅ Principle VIII - Documentation as an API Surface**: Creating `docs/architecture/configuration.md` and linking from central docs index satisfies the requirement that "Every new public surface... MUST have a discoverable entry in the central docs index."

**✅ Principle IX - Test Coverage for Public Behavior**: Deprecation warnings will be tested to ensure they emit correctly and all tests pass (non-breaking).

### Gate Evaluation

**PASS** - No violations detected:
- This is a documentation + deprecation feature that enhances existing constitution compliance
- No new public API surfaces are created (only documenting existing ones)
- Backward compatibility is preserved (non-breaking deprecation warnings)
- Tests will validate deprecation behavior
- Documentation will be properly indexed and discoverable

**No complexity justification needed** - this is a maintenance/documentation task aligned with constitution principles.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
docs/
├── architecture/
│   └── configuration.md         # NEW: Central config hierarchy documentation
├── README.md                     # MODIFIED: Add link to architecture/configuration.md
└── dev_guide.md                  # MODIFIED: Reference new configuration docs

robot_sf/
└── gym_env/
    ├── unified_config.py         # EXISTING: Canonical config classes (documented)
    ├── env_config.py             # MODIFIED: Add deprecation warnings to EnvSettings, PedEnvSettings
    └── sim/
        └── sim_config.py         # NO CHANGE: SimulationSettings is canonical (used by unified_config)

tests/
└── test_gym_env/
    └── test_config_deprecation.py  # NEW: Test deprecation warnings for legacy classes

configs/
├── scenarios/                    # EXISTING: YAML scenario configs (document in new docs)
└── baselines/                    # EXISTING: YAML baseline configs (document in new docs)
```

**Structure Decision**: This is a documentation-focused feature with minimal code changes. The new `docs/architecture/configuration.md` will serve as the central reference for configuration hierarchy, precedence rules, and migration guidance. Legacy config classes in existing modules will be modified to emit deprecation warnings. No new modules are created; only documentation and deprecation annotations are added.

## Complexity Tracking

> **No violations to justify** - All constitution checks passed. This is a documentation and maintenance feature that enhances existing compliance with constitution principles.

## Post-Design Constitution Check

**Re-evaluation Date**: 2025-11-11  
**Status**: ✅ PASS - No violations introduced

### Design Artifacts Review

**Created Artifacts**:
1. `research.md` - Decisions and best practices for config hierarchy documentation
2. `data-model.md` - Configuration class hierarchy and relationships
3. `contracts/test-requirements.md` - Test contracts for deprecation warnings and backward compatibility
4. `quickstart.md` - Implementation guide for developers and migration scenarios for users

**Constitution Alignment**:

**✅ Principle IV - Unified Configuration & Deterministic Seeds**: 
- Documentation reinforces the unified configuration layer
- Deprecation warnings guide users away from fragmented legacy configs
- No changes to seed handling or configuration precedence logic

**✅ Principle VII - Backward Compatibility & Evolution Gates**:
- Non-breaking deprecation warnings preserve all existing functionality
- Migration guide provides clear upgrade path
- Legacy classes remain functional indefinitely (deferred removal to Phase 3)

**✅ Principle VIII - Documentation as an API Surface**:
- New `docs/architecture/configuration.md` will be linked from central index
- Comprehensive coverage of precedence, modules, migration, and best practices
- Code examples validate correct usage patterns

**✅ Principle IX - Test Coverage for Public Behavior**:
- Test contracts defined for deprecation warning emission
- Backward compatibility verification (full test suite must pass)
- Optional config precedence tests for extended validation

**✅ Principle XI - Library Reuse & Helper Documentation**:
- Documentation clarifies existing helper structure (factory functions, config classes)
- No new helpers introduced, only documentation of existing patterns

**No New Violations**: Design phase completed without introducing constitution conflicts. All principles remain satisfied or enhanced.

---

## Implementation Readiness

**Prerequisites Met**:
- ✅ Technical context fully specified (Python 3.11+, pytest, filesystem)
- ✅ All NEEDS CLARIFICATION items resolved in research.md
- ✅ Design artifacts complete (data-model, contracts, quickstart)
- ✅ Constitution compliance verified (pre and post design)
- ✅ Agent context updated with new technologies

**Next Steps**:
1. Run `/speckit.tasks` to generate detailed task breakdown
2. Execute tasks following TDD approach (tests before implementation)
3. Verify quality gates before PR (ruff, type check, pytest)

**Estimated Implementation Scope**:
- Documentation: ~300-500 lines in `configuration.md`
- Code changes: ~20-30 lines (deprecation warnings in 4 classes)
- Tests: ~50-80 lines in `test_config_deprecation.py`
- Doc updates: ~10-20 lines (README.md, dev_guide.md links)

**Risk Assessment**: LOW
- Non-breaking changes only
- Well-defined scope with clear acceptance criteria
- Comprehensive test coverage planned
- Backward compatibility guaranteed

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
