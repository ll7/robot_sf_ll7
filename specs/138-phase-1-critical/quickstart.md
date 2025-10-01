# Quickstart: Type Checking Validation

## Prerequisites
- Python 3.11+ installed
- uv package manager installed
- Repository cloned with submodules

## Setup
```bash
# Activate environment
uv sync

# Verify type checker installation
uvx ty --version
```

## Running Type Checks
```bash
# Run type checking
uvx ty check . --exit-zero

# Expected output: "Found X diagnostics" (target: reduce to <20)
```

## Validation Steps
1. **Check current diagnostics**:
   ```bash
   uvx ty check . --exit-zero | grep "Found" | cut -d' ' -f2
   ```

2. **Verify Python compatibility**:
   ```bash
   python -c "import datetime; print(hasattr(datetime, 'UTC'))"
   # Should print True on Python 3.11+
   ```

3. **Test factory functions**:
   ```bash
   python -c "from robot_sf.gym_env.environment_factory import make_robot_env; env = make_robot_env(); print('Factory works')"
   ```

4. **Run test suite**:
   ```bash
   uv run pytest tests -x
   ```

## Success Criteria
- Type diagnostics reduced to <20 (from 103)
- All factory functions have proper type annotations
- No import resolution errors
- Test suite passes
- Backward compatibility maintained

## Troubleshooting
- If `uvx ty` fails: Install with `pip install uv`
- If import errors: Check Python version compatibility
- If factory errors: Verify config parameters are correctly typed