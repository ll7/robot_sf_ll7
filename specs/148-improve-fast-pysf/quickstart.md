# Quickstart: fast-pysf Integration Quality Improvements

**Feature**: 148-improve-fast-pysf  
**Branch**: `148-improve-fast-pysf`  
**For**: Developers working on fast-pysf integration or running unified tests

## Overview

This guide helps you quickly get started with the improved fast-pysf integration. After this feature is implemented, you'll be able to run robot_sf and fast-pysf tests together, apply unified quality checks, and contribute fixes for PR #236 review comments.

## Prerequisites

✅ Python 3.13.1 (project standard)  
✅ uv package manager installed (`pip install uv`)  
✅ Repository cloned: `git clone https://github.com/ll7/robot_sf_ll7.git`  
✅ Branch checked out: `git checkout 148-improve-fast-pysf`

## Quick Setup (30 seconds)

```bash
# 1. Navigate to repository
cd robot_sf_ll7

# 2. Install dependencies
uv sync --all-extras

# 3. Activate virtual environment (optional, uv run works without this)
source .venv/bin/activate

# 4. Verify installation
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; print('✅ Import successful')"
```

## Running Tests

### Unified Test Suite (Both robot_sf + fast-pysf)

```bash
# Run all tests (recommended)
uv run pytest

# Expected output:
# ======================== test session starts =========================
# collected 55 items (43 robot_sf + 12 fast-pysf)
# ...
# ===================== 55 passed in ~12s =========================
```

### Selective Test Execution

```bash
# Only robot_sf tests
uv run pytest tests

# Only fast-pysf tests
uv run pytest fast-pysf/tests

# Specific test file
uv run pytest fast-pysf/tests/test_forces.py

# Specific test function
uv run pytest fast-pysf/tests/test_forces.py::test_desired_force

# Skip slow tests
uv run pytest -m "not slow"
```

### Headless Mode (for CI or servers)

```bash
# Disable display/graphics
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest
```

### Parallel Execution (faster on multi-core)

```bash
# Auto-detect number of cores
uv run pytest -n auto

# Specific worker count
uv run pytest -n 4
```

## Code Quality Checks

### Formatting and Linting (Ruff)

```bash
# Check for issues (both robot_sf and fast-pysf)
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Combined: fix + format + check
uv run ruff check --fix . && uv run ruff format . && uv run ruff check .
```

### Type Checking

```bash
# Run type checker (ty/pyright)
uvx ty check .

# Exit with zero even if errors found (useful for CI)
uvx ty check . --exit-zero

# Check specific directory
uvx ty check fast-pysf/pysocialforce
```

### Code Quality Task (VS Code)

```bash
# Use predefined task (runs ruff + pylint)
# In VS Code: Cmd+Shift+P → "Tasks: Run Task" → "Check Code Quality"
# Or via terminal:
uv run ruff check . && uv run pylint robot_sf --errors-only
```

## Coverage Analysis

### Generate Coverage Report

```bash
# Run tests with coverage (automatic via pytest config)
uv run pytest

# Coverage is collected automatically; check reports:
# - Terminal: summary printed after test run
# - HTML: htmlcov/index.html
# - JSON: coverage.json
```

### View Coverage Report

```bash
# Open HTML report (macOS)
open htmlcov/index.html

# Or use VS Code task:
# "Tasks: Run Task" → "Open Coverage Report"
```

### Coverage Thresholds

- **robot_sf**: Target ≥ 91% (current baseline)
- **fast-pysf**: Target ≥ 70% (new requirement)

## Working with PR #236 Review Comments

### View Review Comments

```bash
# Read analysis document
cat specs/148-improve-fast-pysf/pr236_review_comments.md

# Or open in editor
code specs/148-improve-fast-pysf/pr236_review_comments.md
```

### Resolve a Comment

1. **Find the issue** in `pr236_review_comments.md` (e.g., #1: Unreachable print)
2. **Locate the file**: `fast-pysf/pysocialforce/map_config.py:81`
3. **Fix the issue**: Remove unreachable code or fix logic
4. **Verify fix**:
   ```bash
   uv run ruff check fast-pysf/pysocialforce/map_config.py
   uv run pytest fast-pysf/tests  # If related test exists
   ```
5. **Commit with reference**:
   ```bash
   git add fast-pysf/pysocialforce/map_config.py
   git commit -m "fix: Remove unreachable print in map_config (#PR236-1)"
   ```
6. **Update tracking**: Mark comment as "Resolved" in `pr236_review_comments.md`

## Common Workflows

### Daily Development

```bash
# 1. Pull latest changes
git pull origin 148-improve-fast-pysf

# 2. Install/update dependencies
uv sync --all-extras

# 3. Run quality gates (before pushing)
uv run ruff check --fix .
uv run ruff format .
uvx ty check . --exit-zero
uv run pytest

# 4. Check coverage
open htmlcov/index.html
```

### Adding a New Test

```bash
# 1. Create test file (if needed)
touch tests/test_new_feature.py

# 2. Write test
cat > tests/test_new_feature.py << 'EOF'
def test_something():
    assert True  # Replace with actual test
EOF

# 3. Run new test
uv run pytest tests/test_new_feature.py -v

# 4. Verify coverage
uv run pytest --cov-report=term-missing tests/test_new_feature.py
```

### Fixing a fast-pysf Test Failure

```bash
# 1. Identify failing test
uv run pytest fast-pysf/tests -v
# Example failure: test_load_map - FileNotFoundError

# 2. Inspect test code
cat fast-pysf/tests/test_map_loader.py

# 3. Check fixture exists
ls fast-pysf/tests/test_maps/
# If missing: create fixture (see Test Fixtures section below)

# 4. Re-run test
uv run pytest fast-pysf/tests/test_map_loader.py::test_load_map -v

# 5. Commit fix
git add fast-pysf/tests/test_maps/map_regular.json
git commit -m "test: Add missing map fixture for test_load_map"
```

## Test Fixtures

### Creating Map Fixtures

**Location**: `fast-pysf/tests/test_maps/`

**Valid map fixture** (`map_regular.json`):
```json
{
  "obstacles": [
    [5.0, 5.0],
    [10.0, 10.0]
  ],
  "routes": [
    {
      "id": "route1",
      "waypoints": [[0.0, 0.0], [20.0, 20.0]]
    }
  ],
  "crowded_zones": [
    {
      "center": [10.0, 10.0],
      "radius": 5.0
    }
  ]
}
```

**Invalid map fixture** (`invalid_json_file.json`):
```json
{
  "obstacles": "not_an_array",
  "routes": [1, 2, 3]
}
```

**Create fixtures**:
```bash
# Create test_maps directory
mkdir -p fast-pysf/tests/test_maps

# Create valid fixture
cat > fast-pysf/tests/test_maps/map_regular.json << 'EOF'
{
  "obstacles": [[5.0, 5.0], [10.0, 10.0]],
  "routes": [{"id": "route1", "waypoints": [[0.0, 0.0], [20.0, 20.0]]}],
  "crowded_zones": [{"center": [10.0, 10.0], "radius": 5.0}]
}
EOF

# Create invalid fixture
cat > fast-pysf/tests/test_maps/invalid_json_file.json << 'EOF'
{
  "obstacles": "not_an_array"
}
EOF
```

## Troubleshooting

### Issue: "No module named 'robot_sf'"

**Solution**:
```bash
# Ensure dependencies installed
uv sync --all-extras

# Or activate venv and reinstall
source .venv/bin/activate
uv sync
```

### Issue: "pytest: command not found"

**Solution**:
```bash
# Use uv run prefix
uv run pytest

# Or activate venv
source .venv/bin/activate
pytest
```

### Issue: "Coverage warning: No data was collected"

**Solution**:
```bash
# Check coverage config in pyproject.toml
# Ensure source includes both packages:
# source = ["robot_sf", "fast-pysf/pysocialforce"]

# Run with explicit coverage
uv run pytest --cov=robot_sf --cov=fast-pysf/pysocialforce
```

### Issue: "ruff: Unknown rule code 'XXX'"

**Solution**:
```bash
# Check ruff version
uv run ruff --version

# Update ruff if needed
uv add --upgrade ruff

# Or check rule code spelling in pyproject.toml
```

### Issue: Tests fail in headless mode

**Solution**:
```bash
# Ensure all environment variables set
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest

# Or use wrapper script
./scripts/validation/test_basic_environment.sh
```

## Performance Expectations

| Operation | Expected Duration | Notes |
|-----------|------------------|-------|
| Full test suite | < 15 seconds | 55 tests total |
| fast-pysf tests only | < 60 seconds | 12 tests (SC-002) |
| robot_sf tests only | ~10 seconds | 43 tests (baseline) |
| Ruff check | < 5 seconds | Both codebases |
| Type check | 10-20 seconds | First run slower (cache) |
| Coverage generation | +2-3 seconds | Adds to test runtime |

## Next Steps

✅ **You're ready!** Start running tests, fixing issues, or contributing improvements.

📚 **Learn More**:
- Feature specification: `specs/148-improve-fast-pysf/spec.md`
- Implementation plan: `specs/148-improve-fast-pysf/implementation_plan.md`
- PR review comments: `specs/148-improve-fast-pysf/pr236_review_comments.md`
- Development guide: `docs/dev_guide.md`

🐛 **Found an Issue?** Create an issue in GitHub or discuss in the team channel.

💡 **Contributing**: See `docs/dev_guide.md` for contribution guidelines.
