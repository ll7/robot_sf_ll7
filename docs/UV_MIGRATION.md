# uv Migration

Migration to the uv package manager for faster dependency resolution and modern Python packaging.

## Migration Complete ✅

2025-06-20

The migration has been successfully completed! Here's what was accomplished:

### ✅ Successfully Migrated
- **Main Project**: All dependencies consolidated into `pyproject.toml`
- **Fast-PySF**: Converted to modern pyproject.toml and integrated as editable dependency
- **Build System**: Switched to `hatchling` backend
- **Documentation**: Updated all README files, CI workflows, and development documentation
- **Scripts & Automation**: Updated all scripts, tasks, and automation to use `uv`
- **Legacy Files**: Removed `requirements.txt` files and legacy `setup.py` files

### ✅ Testing Status
- All 170 tests passing
- Package imports working correctly
- Development workflow functional

### ✅ Files Updated
- `pyproject.toml` - Consolidated main dependencies and configuration
- `fast-pysf/pyproject.toml` - Modernized subproject configuration
- `uv.toml` - Workspace-level uv configuration  
- `README.md` - Updated installation and development instructions
- `.github/workflows/ci.yml` - Updated CI to use uv
- `fast-pysf/.github/workflows/ci.yml` - Updated subproject CI
- `.vscode/tasks.json` - Updated VS Code tasks
- `scripts/update_deps.sh` - Helper script for dependency updates
- Various documentation files and scripts

### ✅ Legacy Files Removed
- `requirements.txt` - Dependencies now in pyproject.toml
- `fast-pysf/requirements.txt` - Dependencies now in fast-pysf/pyproject.toml  
- `fast-pysf/setup.py` - Configuration now in fast-pysf/pyproject.toml

The repository is now fully modernized with `uv` and ready for efficient Python development!

---

# Migration to UV

This document outlines the migration from the old pip/requirements.txt setup to the modern `uv` workflow.

## What Changed

### Before (Old Setup)
- Separate `requirements.txt` files for main project and fast-pysf
- Manual virtual environment management 
- Mixed dependency specification across setup.py and requirements files
- Complex installation process with multiple steps

### After (New Setup)
- Single workspace with `pyproject.toml` files
- Automatic virtual environment management with `uv`
- Centralized dependency specification
- Single command installation and updates

## Migration Steps

### 1. Install uv (if not already installed)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

### 2. Remove old virtual environment (if exists)

```bash
# Remove old virtual environment
rm -rf .venv
```

### 3. Initialize new uv environment

```bash
# Create and sync environment in one step
uv sync
```

### 4. Activate environment

```bash
# Activate the virtual environment
source .venv/bin/activate
```

## Key Commands

### Development Workflow

```bash
# Install all dependencies (production + development)
uv sync

# Install only production dependencies
uv sync --no-dev

# Install with specific extras
uv sync --extra dev --extra gpu

# Add a new dependency
uv add numpy>=1.26.4

# Add a development dependency
uv add --dev pytest>=8.3.3

# Remove a dependency
uv remove package-name

# Update all dependencies
uv lock --upgrade && uv sync

# Run commands in the environment
uv run pytest tests
uv run python script.py
uv run ruff check .
```

### Running Tasks

```bash
# Run tests
uv run pytest tests
uv run pytest test_pygame  # GUI tests

# Code quality
uv run ruff check .
uv run ruff format .
uv run pylint robot_sf

# Update dependencies
./scripts/update_deps.sh
```

## File Changes

### New Files
- `uv.toml` - Workspace configuration
- `fast-pysf/pyproject.toml` - PySocialForce package configuration
- `scripts/update_deps.sh` - Dependency update script

### Modified Files
- `pyproject.toml` - Updated with all dependencies and workspace config
- `setup.py` - Simplified (kept for compatibility)
- `README.md` - Updated installation instructions
- `.github/workflows/ci.yml` - Updated CI to use uv
- `.vscode/tasks.json` - Updated tasks to use uv

### Legacy Files (can be removed after migration)
- `requirements.txt` - Dependencies now in pyproject.toml
- `fast-pysf/requirements.txt` - Dependencies now in fast-pysf/pyproject.toml
- `fast-pysf/setup.py` - Configuration now in fast-pysf/pyproject.toml

## Benefits of the Migration

1. **Faster Installation**: uv is significantly faster than pip
2. **Better Dependency Resolution**: Proper dependency conflict resolution
3. **Workspace Support**: Manages both main project and fast-pysf subproject
4. **Lock File**: Reproducible builds with uv.lock
5. **Simplified Commands**: Single command for most operations
6. **Modern Tooling**: Following Python packaging best practices

## Troubleshooting

### Virtual Environment Issues
```bash
# Reset environment completely
rm -rf .venv uv.lock
uv sync
```

### Dependency Conflicts
```bash
# Check dependency tree
uv tree

# Force update with conflict resolution
uv lock --upgrade --resolution highest
```

### Platform-specific Issues
```bash
# Force platform-specific resolution
uv lock --python-platform linux
uv lock --python-platform darwin
uv lock --python-platform windows
```

## Backward Compatibility

The old pip-based workflow will continue to work, but we recommend migrating to uv for:
- Better performance
- Improved dependency management
- Modern Python packaging standards
- Better CI/CD integration
