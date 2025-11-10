# Feature Specification: Consolidate Utility Modules

**Feature Branch**: `241-consolidate-utility-modules`  
**Created**: November 10, 2025  
**Status**: Draft  
**Input**: User description: "Consolidate fragmented utility modules (util/utils/common) into single robot_sf/common/ module"  
**Related Issue**: [#241](https://github.com/ll7/robot_sf_ll7/issues/241)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Developer Imports Utilities Consistently (Priority: P1)

As a developer working on the robot_sf codebase, I need to import utility functions and types from a single, predictable location so that I can write code faster without searching multiple directories.

**Why this priority**: This addresses the core navigation problem affecting all 50+ import statements across the codebase. Without this, developers waste time searching for utilities and create inconsistent import patterns.

**Independent Test**: Can be fully tested by attempting to import common utilities (Vec2D, set_global_seed, raise_fatal_with_remedy) from robot_sf.common and verifying IDE autocomplete suggests the correct module.

**Acceptance Scenarios**:

1. **Given** a developer needs to import Vec2D type, **When** they type `from robot_sf.common`, **Then** IDE autocomplete shows types.py module containing Vec2D
2. **Given** a developer needs error handling utilities, **When** they search for "raise_fatal", **Then** IDE finds it in robot_sf.common.errors
3. **Given** a developer needs seed utilities, **When** they type `from robot_sf.common`, **Then** they find seed.py (not seed_utils.py in a different location)

---

### User Story 2 - Existing Code Continues to Work (Priority: P1)

As a project maintainer, I need all existing code (tests, examples, scripts) to continue functioning after the consolidation so that the migration doesn't break functionality.

**Why this priority**: This is equally critical as P1 because broken imports would halt all development and testing. The migration must be backward-compatible or comprehensive.

**Independent Test**: Can be fully tested by running the complete test suite (`uv run pytest tests`) and verifying all 893 tests pass after import updates.

**Acceptance Scenarios**:

1. **Given** the utility modules have been consolidated, **When** the test suite runs, **Then** all 893 tests pass without import errors
2. **Given** example scripts exist that import utilities, **When** examples are executed, **Then** they run without ImportError exceptions
3. **Given** the codebase has been migrated, **When** linting runs, **Then** no undefined import errors are reported

---

### User Story 3 - New Contributors Find Utilities Easily (Priority: P2)

As a new contributor to the project, I need clear documentation showing where to find and add utility functions so that I can contribute effectively without creating new fragmentation.

**Why this priority**: Important for long-term maintainability but doesn't block immediate development. Can be added after P1 consolidation is complete.

**Independent Test**: Can be tested by reviewing documentation and verifying it clearly states "all utilities live in robot_sf/common/" with examples.

**Acceptance Scenarios**:

1. **Given** a new contributor reads the dev guide, **When** they search for "where to add utilities", **Then** they find clear guidance pointing to robot_sf/common/
2. **Given** a contributor wants to add a new type alias, **When** they check robot_sf/common/types.py, **Then** they see existing type definitions as examples
3. **Given** documentation exists, **When** searching for import patterns, **Then** examples show imports from robot_sf.common

---

### Edge Cases

- What happens when external code (outside this repository) imports from the old locations (robot_sf.util, robot_sf.utils)?
- How are circular import dependencies handled when consolidating modules?
- What happens if a module contains both public API and internal implementation details?
- How are type checking tools (mypy, pylance) affected by module reorganization?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST consolidate all utility modules into robot_sf/common/ directory
- **FR-002**: System MUST maintain all existing utility functionality without changes to behavior
- **FR-003**: System MUST update all import statements across the codebase to reference robot_sf.common
- **FR-004**: System MUST remove empty utility directories (util/, utils/) after migration
- **FR-005**: System MUST verify all tests pass after migration (893 tests)
- **FR-006**: System MUST preserve backward compatibility through deprecation warnings for a transition period
- **FR-007**: System MUST document the consolidation in CHANGELOG.md with migration guide
- **FR-008**: System MUST ensure IDE autocomplete works correctly with new import paths
- **FR-009**: System MUST rename seed_utils.py to seed.py for consistency
- **FR-010**: System MUST rename compatibility.py to compat.py for brevity

### Key Entities

- **robot_sf.common.types**: Type aliases for geometry (Vec2D, Line2D, Circle2D), robot actions, and poses - provides shared type definitions used across navigation, simulation, and sensor modules
- **robot_sf.common.errors**: Error handling utilities (raise_fatal_with_remedy, warn_soft_degrade) - provides consistent error reporting patterns
- **robot_sf.common.seed**: Random seed management utilities - provides deterministic experiment reproducibility helpers
- **robot_sf.common.compat**: Compatibility layer utilities - provides version-specific compatibility helpers

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All import statements reference robot_sf.common module (100% of ~50 imports updated)
- **SC-002**: Test suite passes with zero import errors (893/893 tests passing)
- **SC-003**: IDE navigation time to find utility functions reduced by 50% (from ~3 clicks through multiple directories to ~1 click)
- **SC-004**: New contributors can locate utility modules in under 30 seconds when following documentation
- **SC-005**: Code search for utility imports returns results from single location (robot_sf.common) rather than 3 different paths
- **SC-006**: Migration completes within 4 hours of developer effort
- **SC-007**: Zero regression bugs reported in production code after deployment

## Scope *(mandatory)*

### In Scope

- Moving util/types.py → common/types.py
- Moving utils/seed_utils.py → common/seed.py (renaming)
- Moving util/compatibility.py → common/compat.py (renaming)
- Updating all import statements in robot_sf/ package
- Updating all import statements in tests/ directory
- Updating all import statements in examples/ directory
- Removing empty util/ and utils/ directories
- Adding migration notes to CHANGELOG.md
- Updating dev_guide.md with new import patterns

### Out of Scope

- Refactoring utility function implementations (only moving, not changing behavior)
- Consolidating utilities from fast-pysf subtree (external dependency)
- Creating new utility functions or removing existing ones
- Changing module interfaces or function signatures
- Performance optimization of utility functions
- Adding type hints to utilities (can be done in separate issue)

## Assumptions *(optional but recommended for complex features)*

1. **No external dependencies**: We assume no external projects depend on direct imports from robot_sf.util or robot_sf.utils (or if they do, they will need to update with version bump)
2. **Automated refactoring is safe**: We assume IDE/editor automated refactoring tools can handle mass import updates correctly
3. **Test coverage is comprehensive**: We assume existing test suite will catch any import-related breakages
4. **Single consolidation is sufficient**: We assume one common/ module is sufficient and won't create its own navigation problems in the future
5. **Module renaming is acceptable**: We assume renaming seed_utils.py → seed.py and compatibility.py → compat.py won't cause confusion

## Dependencies *(optional)*

### Internal Dependencies

- Existing test suite must pass before starting migration
- Ruff linting configuration must be compatible with new import structure
- Type checking tools (mypy, pylance) must recognize new module paths

### External Dependencies

- None - this is an internal refactoring

## Non-Functional Requirements *(optional)*

### Performance

- Import time must not increase (module consolidation should not slow down imports)
- IDE indexing time should improve or remain unchanged

### Maintainability

- Single canonical location for utilities reduces maintenance burden
- Clear module naming (types, errors, seed, compat) improves discoverability
- Removal of duplicate/overlapping modules reduces cognitive load

### Developer Experience

- IDE autocomplete must work seamlessly with new structure
- Error messages from import failures must be clear and helpful
- Migration path must be well-documented for other contributors

## Risks & Mitigations *(optional)*

### Risk 1: Breaking External Code

**Likelihood**: Low  
**Impact**: Medium  
**Mitigation**: 
- Bump minor version (2.0 → 2.1) to signal change
- Add deprecation warnings in stub modules for one release cycle
- Document migration in CHANGELOG with clear before/after examples

### Risk 2: Circular Import Dependencies

**Likelihood**: Low  
**Impact**: High  
**Mitigation**:
- Review dependency graph before moving modules
- Use deferred imports (TYPE_CHECKING blocks) if needed
- Test imports in isolation during migration

### Risk 3: IDE Cache Issues

**Likelihood**: Medium  
**Impact**: Low  
**Mitigation**:
- Document need to restart IDE/clear cache after migration
- Test with fresh virtual environment
- Provide troubleshooting steps in migration notes

### Risk 4: Incomplete Import Updates

**Likelihood**: Low  
**Impact**: High  
**Mitigation**:
- Use automated grep search to find ALL import statements
- Run comprehensive test suite (893 tests)
- Use type checking to catch missing imports
- Manual code review of changes

## Open Questions *(optional)*

*None - specification is complete based on issue #241 details.*
