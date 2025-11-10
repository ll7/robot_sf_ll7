# Quickstart: Utility Module Consolidation Migration

**Feature**: 241-consolidate-utility-modules  
**Audience**: Robot SF developers  
**Time to Complete**: 5-10 minutes  
**Date**: November 10, 2025

## Overview

This guide walks you through migrating from the old fragmented utility structure to the new consolidated `robot_sf.common` module.

**What Changed**:
- ❌ Removed: `robot_sf/util/` directory
- ❌ Removed: `robot_sf/utils/` directory  
- ✅ Consolidated: All utilities now in `robot_sf/common/`
- ✅ Renamed: `seed_utils.py` → `seed.py`, `compatibility.py` → `compat.py`

---

## Step-by-Step Migration

### Step 1: Update Your Branch

```bash
# Pull latest changes
git checkout main
git pull origin main

# If working on a feature branch
git checkout your-feature-branch
git merge main
# or
git rebase main
```

### Step 2: Update Import Statements

**Option A: Manual Find-Replace (Recommended for Small Changes)**

In your code editor (VS Code, PyCharm, etc.):

1. **Find**: `from robot_sf.util.types import`  
   **Replace**: `from robot_sf.common.types import`

2. **Find**: `from robot_sf.util.compatibility import`  
   **Replace**: `from robot_sf.common.compat import`

3. **Find**: `from robot_sf.utils.seed_utils import`  
   **Replace**: `from robot_sf.common.seed import`

**Option B: Automated Script (For Multiple Files)**

```bash
# Navigate to repository root
cd /path/to/robot_sf_ll7

# Run find-replace script
find robot_sf tests examples scripts -name "*.py" -type f -exec sed -i '' \
  -e 's/from robot_sf\.util\.types import/from robot_sf.common.types import/g' \
  -e 's/from robot_sf\.util\.compatibility import/from robot_sf.common.compat import/g' \
  -e 's/from robot_sf\.utils\.seed_utils import/from robot_sf.common.seed import/g' \
  {} +
```

**Note for macOS**: The `-i ''` flag is required for in-place edits. On Linux, use `-i` without quotes.

### Step 3: Verify Changes

```bash
# Check for remaining old imports
grep -r "from robot_sf.util\b" robot_sf/ tests/ examples/ scripts/
grep -r "from robot_sf.utils\b" robot_sf/ tests/ examples/ scripts/

# Expected output: No matches (or only false positives)
```

### Step 4: Run Tests

```bash
# Install dependencies (if needed)
uv sync --all-extras

# Run full test suite
uv run pytest tests

# Expected: 893/893 tests passing
```

### Step 5: Verify Type Checking

```bash
# Run type checker
uvx ty check .

# Expected: 0 type errors (warnings are OK)
```

### Step 6: Run Linter

```bash
# Check for undefined imports
uv run ruff check .

# Expected: No import-related errors
```

---

## Common Migration Patterns

### Pattern 1: Type Imports

**Before**:
```python
from robot_sf.util.types import Vec2D, Line2D, RobotPose
```

**After**:
```python
from robot_sf.common.types import Vec2D, Line2D, RobotPose
```

**Or (convenience import)**:
```python
from robot_sf.common import Vec2D, Line2D, RobotPose
```

---

### Pattern 2: Seed Management

**Before**:
```python
from robot_sf.utils.seed_utils import set_global_seed
```

**After**:
```python
from robot_sf.common.seed import set_global_seed
```

**Or (convenience import)**:
```python
from robot_sf.common import set_global_seed
```

---

### Pattern 3: Error Handling

**Before**:
```python
from robot_sf.common.errors import raise_fatal_with_remedy
```

**After**:
```python
from robot_sf.common.errors import raise_fatal_with_remedy
# (No change - already in robot_sf.common)
```

---

### Pattern 4: Compatibility Shims

**Before**:
```python
from robot_sf.util.compatibility import get_gym_reset_info
```

**After**:
```python
from robot_sf.common.compat import get_gym_reset_info
```

---

## Troubleshooting

### Issue 1: ModuleNotFoundError

**Error**:
```
ModuleNotFoundError: No module named 'robot_sf.util'
```

**Solution**:
1. You missed updating an import statement
2. Run grep verification (Step 3 above)
3. Update any remaining old imports

---

### Issue 2: IDE Autocomplete Not Working

**Problem**: VS Code/PyCharm doesn't suggest `robot_sf.common` imports

**Solution**:
```bash
# Restart Python language server (VS Code)
# Command Palette → "Python: Restart Language Server"

# Or restart IDE
```

---

### Issue 3: Type Checker Complains About Missing Module

**Error**:
```
Cannot find implementation or library stub for module named 'robot_sf.util.types'
```

**Solution**:
1. Clear type checker cache: `rm -rf .mypy_cache/`
2. Re-run type checker: `uvx ty check .`

---

### Issue 4: Import Works Locally But Fails in CI

**Problem**: Tests pass locally but fail in GitHub Actions

**Possible Causes**:
1. Didn't commit import changes: `git add .` and `git commit`
2. Stale virtual environment in CI: Will auto-resolve on next run
3. Cached dependencies: Clear cache in GitHub Actions settings

---

## IDE-Specific Tips

### VS Code

**Automatic Import Updates**:
1. Install Python extension
2. Right-click old import → "Organize Imports"
3. VS Code may auto-update some imports

**Manual Fix**:
1. Open file with old import
2. VS Code will underline missing module
3. Hover → "Quick Fix" → Update import path

---

### PyCharm

**Automatic Refactoring**:
1. Right-click `robot_sf/common/` folder
2. "Refactor" → "Move"
3. PyCharm updates imports automatically

**Note**: This migration already happened, so use manual find-replace instead.

---

### Vim/Neovim

**Bulk Replace**:
```vim
" Open all Python files
:args robot_sf/**/*.py tests/**/*.py examples/**/*.py

" Replace imports
:argdo %s/from robot_sf\.util\.types import/from robot_sf.common.types import/ge | update

" Repeat for other imports
:argdo %s/from robot_sf\.utils\.seed_utils import/from robot_sf.common.seed import/ge | update
```

---

## Validation Checklist

Before considering your migration complete:

- [ ] All old imports replaced (grep verification passes)
- [ ] Tests pass: `uv run pytest tests` (893/893)
- [ ] Type checking passes: `uvx ty check .`
- [ ] Linting passes: `uv run ruff check .`
- [ ] Code runs without import errors
- [ ] Committed changes to git

---

## Quick Reference

### New Import Locations

| Module | New Import Path |
|--------|----------------|
| Types | `from robot_sf.common.types import Vec2D` |
| Errors | `from robot_sf.common.errors import raise_fatal_with_remedy` |
| Seeds | `from robot_sf.common.seed import set_global_seed` |
| Compat | `from robot_sf.common.compat import get_gym_reset_info` |

### Deleted Directories

- `robot_sf/util/` ❌ Removed
- `robot_sf/utils/` ❌ Removed

### Renamed Modules

- `seed_utils.py` → `seed.py`
- `compatibility.py` → `compat.py`

---

## Next Steps

After completing migration:

1. **Review Changes**: `git diff` to verify all updates
2. **Run Quality Gates**: See `docs/dev_guide.md` for full checklist
3. **Commit**: `git commit -m "refactor: migrate to consolidated robot_sf.common imports"`
4. **Optional**: Update any documentation that references old paths

---

## Getting Help

If you encounter issues not covered here:

1. Check the full plan: `specs/241-consolidate-utility-modules/plan.md`
2. Review API contract: `specs/241-consolidate-utility-modules/contracts/api-contract.md`
3. Open a GitHub issue with:
   - Error message (full traceback)
   - Steps to reproduce
   - Output of `grep -r "from robot_sf.util" .`

---

## Summary

**What You Learned**:
- ✅ How to update import statements (manual and automated)
- ✅ How to verify migration completeness
- ✅ Common troubleshooting steps
- ✅ IDE-specific tips

**Time Investment**: 5-10 minutes for typical feature branch

**Outcome**: Clean, consistent imports from single `robot_sf.common` module
