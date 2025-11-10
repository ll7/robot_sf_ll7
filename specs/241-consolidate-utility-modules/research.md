# Research: Consolidate Utility Modules

**Feature**: 241-consolidate-utility-modules  
**Date**: November 10, 2025  
**Status**: Complete

## Overview

This document consolidates research findings for safely consolidating three fragmented utility module directories into a single canonical location.

## Research Tasks

### 1. Python Module Reorganization Best Practices

**Decision**: Use `git mv` for file moves to preserve history + comprehensive grep-based import updates

**Rationale**:
- `git mv` preserves file history in version control (important for future `git blame` and debugging)
- Python's import system doesn't care about file history, only current paths
- Comprehensive search-and-replace of imports ensures no orphaned references
- pytest will catch any missed imports immediately

**Alternatives Considered**:
- Manual copy/paste: Loses git history
- IDE automated refactoring: May miss files outside IDE's scope (scripts, configs)
- Import redirects in old locations: Adds technical debt and doesn't solve navigation problem

**Implementation Approach**:
1. Use `git mv util/types.py common/types.py`
2. Use `git mv utils/seed_utils.py common/seed.py`  
3. Use `git mv util/compatibility.py common/compat.py`
4. Update `common/__init__.py` to export commonly-used symbols
5. Run grep to find all import statements: `grep -r "from robot_sf.util" robot_sf/ tests/ examples/`
6. Use find-replace (manual or scripted) to update imports
7. Run full test suite to verify

### 2. Import Statement Migration Patterns

**Decision**: Direct import path replacement without compatibility shims

**Rationale**:
- This is an internal refactoring affecting only robot_sf codebase
- All import updates happen in a single atomic PR
- Compatibility shims add complexity and don't solve the navigation problem
- Version bump (MINOR: 2.0 → 2.1) signals change to any external consumers

**Alternatives Considered**:
- Deprecation warnings with shims: Adds code complexity, doesn't prevent confusion
- Gradual migration: Creates inconsistent state during transition
- Re-exports from old locations: Hides the consolidation from users, defeats purpose

**Implementation Approach**:
```python
# BEFORE
from robot_sf.util.types import Vec2D
from robot_sf.utils.seed_utils import set_global_seed
from robot_sf.common.errors import raise_fatal_with_remedy

# AFTER
from robot_sf.common.types import Vec2D
from robot_sf.common.seed import set_global_seed
from robot_sf.common.errors import raise_fatal_with_remedy
```

### 3. Circular Import Risk Assessment

**Decision**: No circular import risk - utilities are leaf modules

**Rationale**:
- Examined current import graph: `types.py`, `errors.py`, `seed_utils.py`, `compatibility.py` are all leaf modules
- They don't import from other robot_sf modules (except minimal typing imports)
- Moving them to common/ doesn't create new dependencies
- Test suite will immediately reveal any circular imports

**Alternatives Considered**:
- Detailed dependency analysis tool: Overkill for 4 small utility files
- Deferred imports with TYPE_CHECKING: Not needed, no circular dependencies exist

**Verification**:
```bash
# Check what each utility imports
grep "^from robot_sf" robot_sf/util/types.py robot_sf/utils/seed_utils.py robot_sf/common/errors.py robot_sf/util/compatibility.py
# Expected: None or only TYPE_CHECKING imports
```

### 4. Test Coverage Verification Strategy

**Decision**: Rely on existing comprehensive test suite (893 tests)

**Rationale**:
- Existing test suite already covers all modules that import utilities
- Import errors will cause immediate test failures (explicit, not silent)
- Type checking (ty/mypy) will catch undefined imports
- Linting (Ruff) will catch unused imports

**Test Execution Plan**:
1. Before migration: Run `uv run pytest tests` (establish baseline: 893/893 passing)
2. After file moves: Run `uv run pytest tests` (should fail with import errors)
3. After import updates: Run `uv run pytest tests` (should return to 893/893 passing)
4. Run type check: `uvx ty check .`
5. Run linting: `uv run ruff check .`
6. Run GUI tests: `uv run pytest test_pygame` (if display available)

**Alternatives Considered**:
- Writing new tests for import paths: Redundant, existing tests already verify behavior
- Manual smoke testing only: Insufficient, could miss edge cases

### 5. Module Renaming Conventions

**Decision**: Rename `seed_utils.py` → `seed.py` and `compatibility.py` → `compat.py`

**Rationale**:
- Shorter names improve import brevity: `from robot_sf.common.seed` vs `from robot_sf.common.seed_utils`
- Consistency: Other modules don't use `_utils` suffix (e.g., `types.py` not `type_utils.py`)
- `compat` is a well-understood abbreviation in Python community (e.g., `six.moves.compat`)
- Module contents don't change, only filename

**Naming Justification**:
- `types.py` → `types.py` (already concise)
- `errors.py` → `errors.py` (already concise)
- `seed_utils.py` → `seed.py` (remove redundant `_utils`)
- `compatibility.py` → `compat.py` (common abbreviation)

**Alternatives Considered**:
- Keep `_utils` suffix: Verbose, inconsistent with other modules
- Use `seeds.py` (plural): Confusing (seed management is singular concept)

### 6. IDE and Type Checker Compatibility

**Decision**: Standard Python module moves are fully compatible with modern IDEs and type checkers

**Rationale**:
- VS Code with Pylance automatically indexes new module locations on save
- mypy and ty follow standard Python import resolution
- Ruff linter adapts to new import paths immediately
- May need to restart IDE or clear cache, but this is standard practice

**User Guidance**:
- Document in quickstart.md: "After pulling this change, restart your IDE or clear its cache"
- Add troubleshooting note: "If autocomplete doesn't work, try: VS Code → Command Palette → Python: Restart Language Server"

**Verification**:
- Test in fresh virtual environment to ensure no stale cache dependencies
- Verify autocomplete works with new paths in VS Code

### 7. Documentation Update Requirements

**Decision**: Update dev_guide.md, CHANGELOG.md, and create migration quickstart

**Documentation Updates**:

1. **CHANGELOG.md** (version 2.1.0):
```markdown
## [2.1.0] - 2025-11-XX

### Changed
- **[BREAKING for internal imports]** Consolidated utility modules into single `robot_sf/common/` directory
  - Moved `robot_sf/util/types.py` → `robot_sf/common/types.py`
  - Moved `robot_sf/utils/seed_utils.py` → `robot_sf/common/seed.py` (renamed)
  - Moved `robot_sf/util/compatibility.py` → `robot_sf/common/compat.py` (renamed)
  - Removed empty `robot_sf/util/` and `robot_sf/utils/` directories
  
### Migration Guide
If you have external code importing from `robot_sf.util` or `robot_sf.utils`, update:
- `from robot_sf.util.types import X` → `from robot_sf.common.types import X`
- `from robot_sf.utils.seed_utils import X` → `from robot_sf.common.seed import X`
- `from robot_sf.util.compatibility import X` → `from robot_sf.common.compat import X`
```

2. **docs/dev_guide.md** - Add section on utility module location:
```markdown
### Utility Modules

All shared utility functions and type definitions live in `robot_sf/common/`:
- `robot_sf/common/types` - Type aliases (Vec2D, Line2D, RobotPose, etc.)
- `robot_sf/common/errors` - Error handling utilities
- `robot_sf/common/seed` - Random seed management for reproducibility
- `robot_sf/common/compat` - Compatibility helpers

Example imports:
\`\`\`python
from robot_sf.common.types import Vec2D, RobotPose
from robot_sf.common.errors import raise_fatal_with_remedy
from robot_sf.common.seed import set_global_seed
\`\`\`
```

## Summary

All research tasks complete. Key findings:

1. ✅ Use `git mv` for file moves to preserve history
2. ✅ Direct import path replacement (no shims)
3. ✅ No circular import risks identified
4. ✅ Existing 893-test suite provides sufficient coverage
5. ✅ Rename `seed_utils.py` → `seed.py` and `compatibility.py` → `compat.py`
6. ✅ Standard IDE/type checker compatibility (may need restart)
7. ✅ Documentation updates planned for CHANGELOG and dev_guide

**Ready to proceed to Phase 1: Design & Contracts**
