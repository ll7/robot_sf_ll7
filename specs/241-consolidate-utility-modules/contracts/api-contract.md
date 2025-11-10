# API Contract: robot_sf.common Module

**Feature**: 241-consolidate-utility-modules  
**Version**: 2.1.0  
**Status**: Draft  
**Date**: November 10, 2025

## Contract Overview

This document defines the public API contract for `robot_sf.common` after utility module consolidation. It specifies what imports are guaranteed stable, what's deprecated, and migration guidance for external consumers.

---

## Public API Surface

### Stable Imports (Guaranteed)

The following imports are **guaranteed stable** and will not change in future MINOR or PATCH versions:

```python
from robot_sf.common.types import (
    Vec2D,           # Type alias: NDArray[np.float64] for 2D vectors
    Line2D,          # Type alias: tuple[Vec2D, Vec2D] for line segments
    RobotPose,       # Type alias: tuple[Vec2D, float] for robot state
    normalize_angle, # Function: normalize angle to [-π, π]
)

from robot_sf.common.errors import (
    raise_fatal_with_remedy,  # Function: raise exception with remediation
    ConfigurationError,        # Exception: configuration validation errors
)

from robot_sf.common.seed import (
    set_global_seed,        # Function: set numpy/random/torch seeds
    get_random_state,       # Function: capture RNG state
    restore_random_state,   # Function: restore RNG state
)

from robot_sf.common.compat import (
    get_gym_reset_info,     # Function: check Gym vs Gymnasium API
    wrap_gymnasium_env,     # Function: add compatibility layer
)
```

### Convenience Imports (Recommended)

For commonly-used symbols, `robot_sf.common.__init__.py` re-exports:

```python
from robot_sf.common import (
    Vec2D,
    RobotPose,
    Line2D,
    normalize_angle,
    set_global_seed,
    raise_fatal_with_remedy,
    ConfigurationError,
    get_gym_reset_info,
)
```

**Benefit**: Shorter import statements without obscuring module organization.

---

## Deprecated Imports (Removed in 2.1.0)

The following import paths are **no longer valid** as of version 2.1.0:

```python
# ❌ REMOVED - util/ directory deleted
from robot_sf.util.types import Vec2D
from robot_sf.util.compatibility import get_gym_reset_info

# ❌ REMOVED - utils/ directory deleted  
from robot_sf.utils.seed_utils import set_global_seed

# ❌ REMOVED - module renamed
from robot_sf.common.seed_utils import set_global_seed  # Now: .seed
from robot_sf.common.compatibility import get_gym_reset_info  # Now: .compat
```

**Migration Required**: Update to new import paths (see Migration Guide below).

---

## Semantic Versioning Contract

This consolidation follows semantic versioning:

- **MAJOR**: Not changed (API behavior unchanged)
- **MINOR**: Bumped to 2.1.0 (internal reorganization, import paths changed)
- **PATCH**: Not applicable

**Rationale**: Import path changes are considered MINOR (not MAJOR) because:
1. This is an internal-facing refactoring (robot_sf codebase)
2. External consumers are rare (research project, not public library)
3. Migration is mechanical (find-replace) with no behavioral changes
4. If external consumers exist, CHANGELOG provides clear migration guidance

---

## Migration Guide

### For Internal robot_sf Code

**Automated Migration**:
```bash
# Find all old imports
grep -r "from robot_sf.util" robot_sf/ tests/ examples/ scripts/
grep -r "from robot_sf.utils" robot_sf/ tests/ examples/ scripts/

# Replace with new paths (use editor's find-replace or script)
# See quickstart.md for step-by-step guide
```

**Expected Changes**: ~50 files across:
- `robot_sf/**/*.py` (library code)
- `tests/**/*.py` (unit/integration tests)
- `examples/**/*.py` (demo scripts)
- `scripts/**/*.py` (training/benchmarking)

### For External Consumers (if any)

**Step 1**: Update dependency version
```toml
# pyproject.toml or requirements.txt
robot-sf >= 2.1.0  # or robot-sf~=2.1
```

**Step 2**: Update import statements
```python
# Before (2.0.x)
from robot_sf.util.types import Vec2D
from robot_sf.utils.seed_utils import set_global_seed

# After (2.1.0+)
from robot_sf.common.types import Vec2D
from robot_sf.common.seed import set_global_seed
```

**Step 3**: Verify with tests
```bash
# Run your test suite
pytest tests/
# Check for import errors (will be explicit if missed)
```

---

## Behavioral Guarantees

### No Functional Changes

This refactoring guarantees:
- ✅ All type aliases remain identical (`Vec2D`, `RobotPose`, etc.)
- ✅ All function signatures unchanged
- ✅ All function behavior unchanged
- ✅ All error messages unchanged
- ✅ All test assertions pass (893/893)

### Only Import Path Changes

The **only** user-visible change:
- ❌ Old import paths no longer work
- ✅ New import paths provide same functionality

---

## Compatibility Policy

### Backward Compatibility

**NOT provided** for import paths (breaking change in MINOR version):
- Old paths (`robot_sf.util.*`, `robot_sf.utils.*`) will raise `ModuleNotFoundError`
- No deprecation warnings (clean break)
- No compatibility shims or redirects

**Rationale**: 
- Internal refactoring for research project (limited external consumers)
- Clean migration preferred over ongoing tech debt from compatibility layer
- CHANGELOG provides clear migration path

### Forward Compatibility

**Guaranteed** for new import paths:
- New paths (`robot_sf.common.*`) will remain stable in all 2.x versions
- Future changes will follow semantic versioning
- Any breaking changes will increment MAJOR version (3.0.0)

---

## Testing Contract

### Pre-Merge Requirements

All of the following **must pass** before merging:

1. **Type Checking**:
   ```bash
   uvx ty check .
   # Expected: 0 type errors (warnings acceptable)
   ```

2. **Linting**:
   ```bash
   uv run ruff check .
   # Expected: No errors, no undefined imports
   ```

3. **Unit/Integration Tests**:
   ```bash
   uv run pytest tests
   # Expected: 893/893 passing (0 failures)
   ```

4. **GUI Tests** (if display available):
   ```bash
   uv run pytest test_pygame
   # Expected: All passing (no import errors)
   ```

5. **Import Verification**:
   ```bash
   grep -r "from robot_sf.util\b" robot_sf/ tests/ examples/ scripts/
   grep -r "from robot_sf.utils\b" robot_sf/ tests/ examples/ scripts/
   # Expected: No matches (all updated)
   ```

6. **Functional Smoke Test**:
   ```bash
   uv run python -c "
   from robot_sf.gym_env.environment_factory import make_robot_env
   from robot_sf.common import Vec2D, set_global_seed
   set_global_seed(42)
   env = make_robot_env()
   env.reset()
   print('✓ All imports OK')
   "
   ```

---

## Rollback Policy

If critical issues arise post-merge:

### Rollback Steps
1. Revert PR via GitHub UI
2. Document issue in GitHub comment
3. Create follow-up issue for proper fix
4. Tag `needs-investigation`

### Criteria for Rollback
- Test suite failures (< 893/893 passing)
- Import errors in CI/CD
- Blocking issues for other developers

**Note**: Rollback should be rare given comprehensive test coverage and validation.

---

## Documentation Requirements

### Updated Documentation

1. **CHANGELOG.md**:
   - Add `[2.1.0]` section with migration guide
   - Document all removed import paths
   - Provide before/after examples

2. **docs/dev_guide.md**:
   - Add "Utility Modules" section
   - Document canonical import patterns
   - Update any outdated examples

3. **README.md** (if needed):
   - Update quick start examples if they reference old paths

4. **specs/241-consolidate-utility-modules/quickstart.md**:
   - Step-by-step migration guide for developers
   - Troubleshooting common issues

---

## Contract Checklist

Before marking this feature complete, verify:

- [ ] All public API imports documented
- [ ] All deprecated imports listed
- [ ] Migration guide tested on at least 3 files
- [ ] Semantic versioning justification documented
- [ ] Testing requirements met (all 6 items)
- [ ] CHANGELOG.md updated with migration notes
- [ ] dev_guide.md updated with new import patterns
- [ ] Rollback policy defined
- [ ] No compatibility shims left in codebase

---

## Questions & Clarifications

### Q: Why not provide compatibility shims?
**A**: Compatibility shims would:
- Add technical debt (code to maintain)
- Delay the inevitable migration
- Obscure the consolidation from users
- Not solve the original navigation problem

Clean break with clear migration guide is preferred.

### Q: What if external projects depend on old paths?
**A**: 
1. This is a research project with limited external consumers
2. CHANGELOG provides clear migration instructions
3. MINOR version bump signals change per semver
4. External consumers can pin to `robot-sf<2.1` if needed

### Q: How do we prevent import regressions?
**A**:
1. Comprehensive grep verification before merge
2. 893-test suite catches import errors immediately
3. Type checker validates all imports
4. Linter catches unused/undefined imports

---

## References

- Spec: `specs/241-consolidate-utility-modules/spec.md`
- Research: `specs/241-consolidate-utility-modules/research.md`
- Data Model: `specs/241-consolidate-utility-modules/data-model.md`
- Plan: `specs/241-consolidate-utility-modules/plan.md`
