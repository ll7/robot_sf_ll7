# Research: Configuration Hierarchy Documentation

**Feature**: 244-document-config-hierarchy  
**Date**: 2025-11-11  
**Status**: Complete

## Overview

This document captures research findings and decisions for documenting the configuration hierarchy and deprecating legacy config classes in robot_sf.

## Decision 1: Configuration Precedence Model

**Decision**: Document a three-tier precedence hierarchy: Code Defaults < YAML Files < Runtime Parameters

**Rationale**:
- **Code Defaults**: Defined in dataclass `field(default=...)` or `field(default_factory=...)` in `unified_config.py` classes
- **YAML Files**: Scenario and baseline configs in `configs/scenarios/` and `configs/baselines/` can override defaults
- **Runtime Parameters**: Factory function kwargs (e.g., `make_robot_env(config=custom_config)`) take final precedence

This model is consistent with established patterns in Python configuration libraries (e.g., Hydra, pydantic-settings) and matches the actual implementation in `environment_factory.py`.

**Alternatives Considered**:
- Environment variables as a tier: Rejected - not currently used in robot_sf for configuration
- Single-tier (runtime only): Rejected - loses discoverability of defaults and scenario reusability
- YAML-first precedence: Rejected - would break existing runtime override patterns

**Evidence**: 
- `environment_factory.py` shows runtime kwargs override config objects
- `unified_config.py` dataclasses define code defaults
- Existing YAML configs in `configs/` demonstrate file-based overrides

## Decision 2: Legacy Config Classes Identification

**Decision**: Mark the following as legacy with deprecation warnings:
- `EnvSettings` in `robot_sf/gym_env/env_config.py`
- `PedEnvSettings` in `robot_sf/gym_env/env_config.py`
- `RobotEnvSettings` in `robot_sf/gym_env/env_config.py` (intermediate class, also legacy)
- `BaseEnvSettings` in `robot_sf/gym_env/env_config.py` (base of legacy hierarchy)

**Canonical Classes** (in `unified_config.py`):
- `BaseSimulationConfig` - Core simulation configuration
- `RobotSimulationConfig` - Robot-based environments
- `ImageRobotConfig` - Robot with image observations
- `PedestrianSimulationConfig` - Pedestrian-adversary environments
- `MultiRobotConfig` - Multi-robot scenarios (if exists)

**Rationale**:
- `unified_config.py` was created explicitly to consolidate and replace the fragmented config classes
- Legacy classes in `env_config.py` duplicate functionality and create confusion
- Module docstrings in `env_config.py` describe the purpose but don't indicate legacy status
- Factory functions in `environment_factory.py` accept `RobotSimulationConfig` and related classes

**Alternatives Considered**:
- Immediate removal: Rejected - breaks backward compatibility
- No deprecation (leave as-is): Rejected - perpetuates confusion and technical debt
- Gradual migration without warnings: Rejected - users won't know to migrate

**Migration Path**:
- `EnvSettings` → `RobotSimulationConfig` (both have robot_config, sim_config, lidar_config, map_pool)
- `PedEnvSettings` → `PedestrianSimulationConfig` (adds ego_ped_config to robot base)
- `RobotEnvSettings` → `RobotSimulationConfig` (direct equivalent)

## Decision 3: Deprecation Warning Strategy

**Decision**: Use Python's `warnings.warn()` with `DeprecationWarning` category in `__init__` methods

**Implementation Pattern**:
```python
import warnings

@dataclass
class EnvSettings:
    def __post_init__(self):
        warnings.warn(
            "EnvSettings is deprecated and will be removed in a future version. "
            "Use RobotSimulationConfig from robot_sf.gym_env.unified_config instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # ... existing validation code ...
```

**Rationale**:
- Non-breaking: warnings don't halt execution
- Standard Python mechanism: users expect `DeprecationWarning`
- `stacklevel=2` points to caller's code, not the config class
- Message includes replacement class for easy migration
- Can be suppressed if needed: `warnings.filterwarnings("ignore", category=DeprecationWarning)`

**Alternatives Considered**:
- Loguru logger warnings: Rejected - deprecations are Python stdlib convention
- Custom exception: Rejected - breaking change
- Decorator-based deprecation: Rejected - unnecessary complexity for dataclasses
- No stack level: Rejected - less helpful to users

## Decision 4: Documentation Structure

**Decision**: Create `docs/architecture/configuration.md` as the central configuration reference

**Sections**:
1. **Overview** - Purpose and scope of configuration system
2. **Precedence Hierarchy** - Code defaults < YAML < Runtime (with examples)
3. **Configuration Modules** - Canonical vs legacy module map
4. **Unified Config Classes** - Description of each class in `unified_config.py`
5. **YAML Configuration** - How scenario/baseline configs map to classes
6. **Runtime Overrides** - Factory function parameter examples
7. **Migration Guide** - Legacy → unified conversions with code examples
8. **External Configuration** - fast-pysf and other external configs
9. **Best Practices** - When to use each configuration level

**Rationale**:
- `architecture/` directory is appropriate for system-level design docs
- Centralized reference reduces duplication across `dev_guide.md`, `ENVIRONMENT.md`, etc.
- Progressive disclosure: overview → details → migration → advanced topics
- Code examples make migration concrete

**Alternatives Considered**:
- Inline in `dev_guide.md`: Rejected - too long, clutters main development guide
- Separate migration doc: Rejected - better to have migration in same doc as reference
- Multiple small docs: Rejected - harder to discover, maintain consistency

## Decision 5: Test Strategy for Deprecation Warnings

**Decision**: Create `tests/test_gym_env/test_config_deprecation.py` to validate warnings

**Test Cases**:
1. Instantiating `EnvSettings` emits `DeprecationWarning` with correct message
2. Instantiating `PedEnvSettings` emits `DeprecationWarning` with correct message
3. Instantiating `RobotEnvSettings` emits `DeprecationWarning` with correct message
4. Instantiating `BaseEnvSettings` emits `DeprecationWarning` with correct message
5. All existing tests pass after adding warnings (backward compatibility check)
6. Warning message includes replacement class name

**Implementation Pattern**:
```python
import warnings
import pytest

def test_env_settings_deprecated():
    with pytest.warns(DeprecationWarning, match="RobotSimulationConfig"):
        from robot_sf.gym_env.env_config import EnvSettings
        EnvSettings()
```

**Rationale**:
- `pytest.warns()` is the standard way to test warnings
- `match` parameter validates message content
- Running full test suite ensures non-breaking change
- Isolated test file keeps deprecation tests organized

**Alternatives Considered**:
- Manual testing only: Rejected - not reproducible, easy to miss regressions
- Inline tests in existing test files: Rejected - pollutes unrelated test modules
- No tests: Rejected - violates Constitution Principle IX (test coverage requirement)

## Best Practices Summary

### For Documentation
1. Use concrete code examples for each migration scenario
2. Link from `docs/README.md` and `dev_guide.md` for discoverability
3. Include both "before" and "after" snippets in migration guide
4. Document edge cases (e.g., mixed legacy/unified usage)
5. Reference constitution principles where applicable

### For Deprecation Warnings
1. Always include replacement class name in warning message
2. Use `stacklevel=2` to point to user code
3. Emit warning in `__post_init__()` after validation succeeds
4. Don't break existing functionality - warnings only
5. Test that warnings appear and all tests pass

### For Migration Guide
1. Group conversions by use case (robot, image, pedestrian)
2. Show parameter mapping (e.g., `robot_config` in both classes)
3. Note behavioral differences if any exist (likely none)
4. Provide complete working examples, not just snippets
5. Link to factory function documentation

## Implementation Checklist

- [x] Identify legacy config classes and canonical replacements
- [x] Define deprecation warning strategy and message format
- [x] Plan documentation structure and sections
- [x] Design test strategy for deprecation warnings
- [ ] Create `docs/architecture/configuration.md` (Phase 1)
- [ ] Add deprecation warnings to legacy classes (Phase 1)
- [ ] Update `docs/README.md` and `dev_guide.md` links (Phase 1)
- [ ] Create deprecation warning tests (Phase 1)
- [ ] Run full test suite to verify backward compatibility (Phase 1)

## References

- **Constitution Principle IV**: Unified Configuration & Deterministic Seeds
- **Constitution Principle VII**: Backward Compatibility & Evolution Gates
- **Constitution Principle VIII**: Documentation as an API Surface
- Existing config modules: `robot_sf/gym_env/unified_config.py`, `robot_sf/gym_env/env_config.py`
- Environment factory: `robot_sf/gym_env/environment_factory.py`
- YAML configs: `configs/scenarios/`, `configs/baselines/`
