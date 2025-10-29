# Git Subtree Migration: fast-pysf

## Overview

This document describes the migration from git submodule to git subtree for the `fast-pysf` integration with the [pysocialforce-ll7](https://github.com/ll7/pysocialforce-ll7) repository.

**Migration Date**: October 29, 2025  
**Branch**: `ll7/replace_submodule_with_subtree`  
**Upstream Repository**: https://github.com/ll7/pysocialforce-ll7  
**Upstream Branch**: `robot-sf`  
**Commit**: `1021476` (102147601ad439597dd40f2beb52cd952db73a0c)

## Motivation

The migration from git submodule to git subtree was undertaken to:

1. **Simplify workflow**: Eliminate the need for `git submodule update --init --recursive` after cloning
2. **Improve discoverability**: Make the fast-pysf code directly visible in the repository
3. **Preserve history**: Keep the complete git history from the upstream repository
4. **Easier CI/CD**: Remove submodule initialization steps from build processes
5. **Better developer experience**: Allow direct edits and commits without submodule complexity

## What Changed

### Before (Submodule)
- `fast-pysf/` was a git submodule pointing to pysocialforce-ll7
- Required explicit initialization: `git submodule update --init --recursive`
- Submodule configuration in `.gitmodules`
- Separate repository checkout within the main repository
- Developers needed to understand submodule workflows

### After (Subtree)
- `fast-pysf/` is now a git subtree with full code and history integrated
- No initialization required after clone
- Remote configured as `pysocialforce` for future updates
- Full pysocialforce code and history merged into robot_sf_ll7
- Developers work with it as regular directory

## Migration Steps

The migration was performed using the following commands:

```bash
# 1. Remove the existing submodule (if present)
git rm fast-pysf
git commit -m "Remove fast-pysf submodule and related configuration"

# 2. Add the upstream repository as a remote
git remote add -f pysocialforce https://github.com/ll7/pysocialforce-ll7.git

# 3. Add the subtree from the robot-sf branch (preserving history)
git subtree add --prefix=fast-pysf pysocialforce robot-sf
```

The result is a merge commit that integrates the complete pysocialforce history at commit `1021476`.

## Directory Structure

After migration, the `fast-pysf/` directory contains:

```
fast-pysf/
├── .github/          # CI workflows
├── .gitignore
├── .pylintrc
├── LICENSE
├── README.md
├── benchmarks/       # Performance benchmarks
├── docs/            # Pysocialforce documentation
├── examples/        # Usage examples
├── maps/            # Map definitions
├── pyproject.toml   # Project configuration
├── pysocialforce/   # Core library code
├── tests/           # Test suite
└── uv.lock          # Dependency lock file
```

## Working with the Subtree

### Regular Development

For most development work, treat `fast-pysf/` as a regular directory:

```bash
# Edit files directly
vim fast-pysf/pysocialforce/simulator.py

# Commit changes normally
git add fast-pysf/
git commit -m "Update simulator logic"
```

### Pulling Updates from Upstream

To sync with updates from the pysocialforce-ll7 repository:

```bash
# Pull changes from the robot-sf branch
git subtree pull --prefix=fast-pysf pysocialforce robot-sf

# Or from a specific commit
git subtree pull --prefix=fast-pysf pysocialforce <commit-hash>
```

### Pushing Changes Upstream (if needed)

If you make changes in `fast-pysf/` that should go back to pysocialforce-ll7:

```bash
# Push subtree changes to upstream
git subtree push --prefix=fast-pysf pysocialforce robot-sf
```

**Note**: Coordinate with the pysocialforce-ll7 maintainers before pushing changes upstream.

### Checking Subtree Status

To see the subtree merge history:

```bash
# View subtree commits
git log --oneline --graph fast-pysf/

# Find subtree merge points
git log --grep="Subtree" --oneline
```

## Integration Points

### Python Imports

The integration remains unchanged for Python code:

```python
from robot_sf.sim.FastPysfWrapper import FastPysfWrapper
# FastPysfWrapper internally uses fast-pysf/pysocialforce
```

### Dependencies

The `fast-pysf/` subtree has its own `pyproject.toml` with dependencies. The main `robot_sf_ll7` project includes these transitively through the FastPysfWrapper integration.

Key dependencies from fast-pysf:
- `numpy>=1.26.4`
- `numba>=0.60.0`
- Additional rendering libraries (pygame, matplotlib, etc.)

### Testing

Tests for fast-pysf code:

```bash
# Run fast-pysf specific tests
uv run python -m pytest fast-pysf/tests/ -v

# Run main project tests (includes FastPysfWrapper integration tests)
uv run pytest tests/
```

## CI/CD Impact

### Before (with submodule)
```yaml
- name: Checkout code
  uses: actions/checkout@v4
  with:
    submodules: recursive  # Required for submodules
```

### After (with subtree)
```yaml
- name: Checkout code
  uses: actions/checkout@v4
  # No submodule initialization needed!
```

## Troubleshooting

### Issue: "couldn't find remote ref"

If you get errors about missing refs when pulling:

```bash
# Ensure the remote is properly configured
git remote -v | grep pysocialforce

# Re-add if needed
git remote add -f pysocialforce https://github.com/ll7/pysocialforce-ll7.git
```

### Issue: Merge conflicts during subtree pull

When pulling updates causes conflicts:

```bash
# Resolve conflicts in fast-pysf/ files
git status  # Check conflicted files

# After resolving
git add fast-pysf/
git commit -m "Merge pysocialforce updates"
```

### Issue: Want to revert to specific upstream version

To sync to a specific commit from pysocialforce-ll7:

```bash
# Fetch the specific commit
git fetch pysocialforce robot-sf

# Pull to that commit (may require manual merge)
git subtree pull --prefix=fast-pysf pysocialforce <commit-hash>
```

## Verification

After the migration, verify the integration:

```bash
# 1. Check directory exists and has content
ls -la fast-pysf/

# 2. Verify Python imports work
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; print('Import successful')"

# 3. Run fast-pysf tests
uv run python -m pytest fast-pysf/tests/ -v

# 4. Run main test suite
uv run pytest tests/
```

## References

- **Git Subtree Documentation**: https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging
- **Upstream Repository**: https://github.com/ll7/pysocialforce-ll7
- **Integration Wrapper**: `robot_sf/sim/FastPysfWrapper.py`
- **Development Guide**: `docs/dev_guide.md`

## Migration Commit

The subtree was added in commit `c872663`:
```
commit c872663
Author: [Your Name]
Date:   Tue Oct 29 09:08:00 2025

    Add 'fast-pysf/' from commit '102147601ad439597dd40f2beb52cd952db73a0c'
```

This commit includes the complete history from pysocialforce-ll7's `robot-sf` branch up to commit `1021476`.

## Future Considerations

### Updating Strategy

Recommended workflow for staying in sync:

1. **Quarterly sync**: Pull updates from pysocialforce-ll7 every 3 months
2. **Tagged releases**: Coordinate with upstream when they tag releases
3. **Critical fixes**: Pull immediately for security or critical bug fixes

### Contributing Back

If making significant improvements to fast-pysf code:

1. Test changes thoroughly in robot_sf_ll7 context
2. Extract changes to separate branch
3. Submit PR to pysocialforce-ll7 repository
4. Pull back merged changes via subtree

### Long-term Maintenance

Consider:
- **Documentation**: Keep this doc updated with any workflow changes
- **Remote management**: The `pysocialforce` remote should be maintained
- **Conflict resolution**: Document common merge conflict patterns
- **Version tracking**: Tag robot_sf_ll7 releases with corresponding fast-pysf versions

---

**Last Updated**: October 29, 2025  
**Maintainer**: robot_sf_ll7 team  
**Related Docs**: `docs/fast_pysf_wrapper.md`, `docs/dev_guide.md`
