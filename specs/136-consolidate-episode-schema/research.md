# Research: Consolidate Episode Schema Definitions

**Date**: 2025-09-26
**Researcher**: GitHub Copilot
**Context**: Feature to establish single source of truth for episode schema definitions

## Research Questions & Findings

### RQ1: Current Schema Duplication Status
**Question**: Where are duplicate episode schemas located and how identical are they?

**Findings**:
- **Primary location**: `robot_sf/benchmark/schemas/episode.schema.v1.json` (121 lines)
- **Duplicate location**: `specs/120-social-navigation-benchmark-plan/contracts/episode.schema.v1.json` (121 lines)
- **Verification**: Files are byte-for-byte identical (confirmed via diff)
- **Impact**: Any schema change requires manual synchronization across both locations

**Decision**: Establish `robot_sf/benchmark/schemas/episode.schema.v1.json` as canonical source
**Rationale**: This location is already referenced by contract tests and appears to be the active implementation location

### RQ2: Git Hook Implementation for Duplicate Prevention
**Question**: How can git hooks prevent creation of new duplicate schema files?

**Findings**:
- **Current hook setup**: Pre-commit framework with Ruff linting/formatting only
- **Hook type needed**: Pre-commit hook to scan for schema file patterns
- **Detection logic**: 
  - Identify JSON files with schema patterns (*.schema.v*.json)
  - Compare file contents using hashing or structural diff
  - Flag commits containing duplicate schemas
- **Implementation approach**: Custom pre-commit hook script in Python

**Decision**: Add custom pre-commit hook `hooks/prevent-schema-duplicates.py`
**Rationale**: Integrates with existing pre-commit framework, written in Python for consistency

### RQ3: Semantic Versioning for Schema Evolution
**Question**: How should schema versioning work with breaking change detection?

**Findings**:
- **Current pattern**: Filename includes version (episode.schema.v1.json), schema has const "v1"
- **Versioning scheme**: Semantic versioning (major.minor.patch)
  - **Major**: Breaking changes (removed/renamed required fields)
  - **Minor**: Backward-compatible additions (new optional fields)
  - **Patch**: Documentation/clarification changes
- **Breaking change detection**: Compare schema structure, not just content
- **Migration strategy**: Support multiple versions during transition periods

**Decision**: Implement semantic versioning with automated breaking change detection
**Rationale**: Aligns with JSON Schema best practices and enables safe evolution

### RQ4: Runtime Resolution Patterns
**Question**: How should code load schemas from canonical location?

**Findings**:
- **Current loading**: Direct file path in contract tests (`robot_sf/benchmark/schemas/episode.schema.v1.json`)
- **Resolution patterns**:
  - **Import-time**: Load schema as module constant
  - **Runtime**: Lazy loading with caching
  - **Validation**: Schema loaded on-demand for validation calls
- **Path resolution**: Use `importlib.resources` or `pathlib.Path` relative to package
- **Error handling**: Clear error messages if canonical schema missing/corrupted

**Decision**: Runtime resolution with importlib.resources for package-relative loading
**Rationale**: Works in packaged distributions, handles virtual environments correctly, provides clear error messages

## Alternatives Considered

### Schema Storage Location
- **Option A**: Keep in `robot_sf/benchmark/schemas/` (chosen)
  - Pro: Already established location, referenced by tests
  - Con: Deep nesting in package structure
- **Option B**: Move to top-level `schemas/` directory
  - Pro: Easier discovery, less nesting
  - Con: Requires updating all import paths and tests

### Versioning Strategy
- **Option A**: Semantic versioning with file naming (chosen)
  - Pro: Clear evolution path, industry standard
  - Con: Requires tooling for breaking change detection
- **Option B**: Date-based versioning (YYYY-MM-DD)
  - Pro: Simple, chronological
  - Con: Doesn't convey compatibility information

### Hook Implementation Language
- **Option A**: Python script (chosen)
  - Pro: Consistent with codebase, easy testing
  - Con: Requires Python environment in pre-commit
- **Option B**: Shell script
  - Pro: No Python dependency
  - Con: Complex JSON parsing in shell, harder to maintain

## Technical Approach Summary

**Architecture**: Single canonical schema file with runtime resolution and git-based enforcement

**Key Components**:
1. **Canonical Schema**: `robot_sf/benchmark/schemas/episode.schema.v1.json`
2. **Resolution Module**: New `robot_sf/benchmark/schema_loader.py` for loading schemas
3. **Prevention Hook**: `hooks/prevent-schema-duplicates.py` in pre-commit config
4. **Version Detection**: Schema comparison utility for semantic versioning

**Migration Strategy**:
1. Establish canonical schema location
2. Update all references to use resolution module
3. Remove duplicate files
4. Add prevention hook
5. Update documentation

**Risk Mitigation**:
- Backward compatibility maintained during transition
- Clear error messages for missing schemas
- Automated testing prevents regressions</content>
<parameter name="filePath">/Users/lennart/git/robot_sf_ll7/specs/136-consolidate-episode-schema/research.md