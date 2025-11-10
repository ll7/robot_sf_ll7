# Data Model: robot_sf.common Module Structure

**Feature**: 241-consolidate-utility-modules  
**Date**: November 10, 2025

## Module Overview

The `robot_sf.common` module provides shared utilities, type definitions, and error handling used throughout the robot_sf package. After consolidation, it contains **5 modules**:

```
robot_sf/common/
├── __init__.py      # Public API exports
├── types.py         # Type aliases and definitions
├── errors.py        # Error handling utilities (existing)
├── seed.py          # Random seed management
└── compat.py        # Compatibility layer for dependencies
```

## Module Responsibilities

### 1. types.py

**Purpose**: Centralized type definitions for geometric primitives and robot state.

**Exports**:
```python
# Type aliases
Vec2D = NDArray[np.float64]          # 2D vector [x, y]
Line2D = tuple[Vec2D, Vec2D]         # Line segment (start, end)
RobotPose = tuple[Vec2D, float]      # Robot state (position, orientation)

# Functions
def normalize_angle(angle: float) -> float
    """Normalize angle to [-π, π] range"""
```

**Dependencies**:
- External: `numpy`, `typing`
- Internal: None (leaf module)

**Usage Patterns**:
- Imported by: `robot_sf/gym_env/`, `robot_sf/sim/`, `robot_sf/nav/`
- Typical use: `from robot_sf.common.types import Vec2D, RobotPose`

---

### 2. errors.py

**Purpose**: Standardized error handling and remediation guidance.

**Exports**:
```python
def raise_fatal_with_remedy(msg: str, remedy: str) -> NoReturn
    """Raise exception with remediation guidance"""

class ConfigurationError(Exception):
    """Raised for invalid configuration values"""
```

**Dependencies**:
- External: `loguru`
- Internal: None (leaf module)

**Usage Patterns**:
- Imported by: `robot_sf/gym_env/`, `robot_sf/sim/`, configuration validation
- Typical use: 
  ```python
  from robot_sf.common.errors import raise_fatal_with_remedy
  if invalid_config:
      raise_fatal_with_remedy("Invalid value", "Set param to X")
  ```

---

### 3. seed.py (renamed from seed_utils.py)

**Purpose**: Global random seed management for reproducible experiments.

**Exports**:
```python
def set_global_seed(seed: int | None) -> None
    """Set seed for numpy, random, and torch (if available)"""

def get_random_state() -> dict[str, Any]
    """Capture current RNG state for checkpointing"""

def restore_random_state(state: dict[str, Any]) -> None
    """Restore RNG state from checkpoint"""
```

**Dependencies**:
- External: `numpy`, `random`, `torch` (optional)
- Internal: None (leaf module)

**Usage Patterns**:
- Imported by: `scripts/training_ppo.py`, test fixtures, benchmark runner
- Typical use:
  ```python
  from robot_sf.common.seed import set_global_seed
  set_global_seed(42)  # Reproducible experiments
  ```

---

### 4. compat.py (renamed from compatibility.py)

**Purpose**: Compatibility shims for Gym/Gymnasium API differences.

**Exports**:
```python
def get_gym_reset_info() -> bool
    """Check if gym.reset() returns info dict (Gymnasium style)"""

def wrap_gymnasium_env(env: gym.Env) -> gym.Env
    """Add compatibility layer for Gymnasium→Gym API"""
```

**Dependencies**:
- External: `gymnasium`, `gym`
- Internal: None (leaf module)

**Usage Patterns**:
- Imported by: `robot_sf/gym_env/environment.py`
- Typical use:
  ```python
  from robot_sf.common.compat import get_gym_reset_info
  if get_gym_reset_info():
      obs, info = env.reset()  # Gymnasium style
  else:
      obs = env.reset()  # Legacy Gym style
  ```

---

### 5. __init__.py

**Purpose**: Public API surface for robot_sf.common module.

**Structure**:
```python
"""
Common utilities for robot_sf package.

This module provides shared type definitions, error handling,
seed management, and compatibility shims.
"""

# Re-export commonly used symbols
from robot_sf.common.types import Vec2D, Line2D, RobotPose, normalize_angle
from robot_sf.common.errors import raise_fatal_with_remedy, ConfigurationError
from robot_sf.common.seed import set_global_seed
from robot_sf.common.compat import get_gym_reset_info

__all__ = [
    # Types
    "Vec2D",
    "Line2D",
    "RobotPose",
    "normalize_angle",
    # Errors
    "raise_fatal_with_remedy",
    "ConfigurationError",
    # Seed management
    "set_global_seed",
    # Compatibility
    "get_gym_reset_info",
]
```

**Design Decision**: Export commonly-used symbols from `__init__.py` to allow both:
- Explicit imports: `from robot_sf.common.types import Vec2D`
- Convenience imports: `from robot_sf.common import Vec2D`

---

## Dependency Graph

```
External dependencies
    ↓
robot_sf/common/ (leaf modules, no internal dependencies)
    ↓
robot_sf/gym_env/, robot_sf/sim/, robot_sf/nav/, robot_sf/render/
    ↓
scripts/, examples/, tests/
```

**Key Property**: All modules in `robot_sf/common/` are **leaf modules** with no internal dependencies on other robot_sf packages. This prevents circular imports and ensures stable imports across the codebase.

---

## Migration Impact

### Modules to Move

| Current Location | New Location | Rename? |
|-----------------|--------------|---------|
| `robot_sf/util/types.py` | `robot_sf/common/types.py` | No |
| `robot_sf/util/compatibility.py` | `robot_sf/common/compat.py` | Yes |
| `robot_sf/utils/seed_utils.py` | `robot_sf/common/seed.py` | Yes |
| `robot_sf/common/errors.py` | `robot_sf/common/errors.py` | No (already in place) |

### Import Statement Changes

**Before**:
```python
from robot_sf.util.types import Vec2D
from robot_sf.util.compatibility import get_gym_reset_info
from robot_sf.utils.seed_utils import set_global_seed
from robot_sf.common.errors import raise_fatal_with_remedy
```

**After**:
```python
from robot_sf.common.types import Vec2D
from robot_sf.common.compat import get_gym_reset_info
from robot_sf.common.seed import set_global_seed
from robot_sf.common.errors import raise_fatal_with_remedy
```

**Estimated Import Updates**: ~50 files across robot_sf/, tests/, examples/, scripts/

---

## Validation Strategy

### 1. Static Analysis
- Type checking: `uvx ty check .` (must pass with 0 errors)
- Linting: `uv run ruff check .` (must pass, no undefined imports)

### 2. Test Execution
- Unit/integration tests: `uv run pytest tests` (893/893 passing)
- GUI tests: `uv run pytest test_pygame` (if display available)
- Fast-pysf subtree: `uv run pytest fast-pysf/tests` (12/12 passing)

### 3. Import Verification
```bash
# Find any remaining old imports
grep -r "from robot_sf.util" robot_sf/ tests/ examples/ scripts/
grep -r "from robot_sf.utils" robot_sf/ tests/ examples/ scripts/
# Expected: No matches (all updated to robot_sf.common)
```

### 4. Functional Smoke Tests
```bash
# Environment creation
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; env = make_robot_env(); env.reset(); print('✓ Environment creation OK')"

# Type imports
uv run python -c "from robot_sf.common.types import Vec2D, RobotPose; print('✓ Type imports OK')"

# Seed management
uv run python -c "from robot_sf.common.seed import set_global_seed; set_global_seed(42); print('✓ Seed utilities OK')"
```

---

## Public API Contract

### Stable Exports (after consolidation)

```python
# Guaranteed stable imports (for external users)
from robot_sf.common import (
    Vec2D,           # Type alias for np.ndarray
    RobotPose,       # Type alias for robot state
    Line2D,          # Type alias for line segments
    set_global_seed, # Seed management function
    raise_fatal_with_remedy,  # Error helper
)
```

### Deprecated Imports (removed)

```python
# ❌ REMOVED (no longer valid after this feature)
from robot_sf.util.types import Vec2D
from robot_sf.utils.seed_utils import set_global_seed
from robot_sf.util.compatibility import get_gym_reset_info
```

---

## Notes

- All modules in `robot_sf/common/` are **pure utility modules** with no side effects on import
- No circular dependencies exist or will be introduced
- Module renaming (`seed_utils` → `seed`, `compatibility` → `compat`) improves consistency with Python community conventions
- `__init__.py` re-exports provide convenience without hiding module organization
