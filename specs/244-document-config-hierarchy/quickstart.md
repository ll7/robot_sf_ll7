# Quickstart: Configuration Hierarchy Documentation

**Feature**: 244-document-config-hierarchy  
**Date**: 2025-01-11

## For Developers: Using the Configuration Documentation

### 1. Find the Configuration Documentation

**Location**: `docs/architecture/configuration.md`

**How to get there**:
- From repository root: Open `docs/README.md` → Look for "Configuration" section → Click link
- Direct path: `docs/architecture/configuration.md`
- From dev guide: `docs/dev_guide.md` → Search for "configuration" → Follow link

### 2. Understand Configuration Precedence

**Quick Reference**:
```
Code Defaults < YAML Files < Runtime Parameters
```

**Example**:
```python
# Code default: sim_time_in_secs = 200.0
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.sim.sim_config import SimulationSettings

# YAML can override to 150.0 (in configs/scenarios/my_scenario.yaml)
# Runtime override wins:
config = RobotSimulationConfig(
    sim_config=SimulationSettings(sim_time_in_secs=100.0)  # This value is used
)
```

### 3. Migrate from Legacy Config Classes

**If you see this warning**:
```
DeprecationWarning: EnvSettings is deprecated and will be removed in a future version. 
Use RobotSimulationConfig from robot_sf.gym_env.unified_config instead.
```

**Quick fix**:
```python
# OLD (legacy - will be removed)
from robot_sf.gym_env.env_config import EnvSettings
config = EnvSettings()

# NEW (canonical - recommended)
from robot_sf.gym_env.unified_config import RobotSimulationConfig
config = RobotSimulationConfig()
```

**See full migration guide**: `docs/architecture/configuration.md` → "Migration Guide" section

---

## For Maintainers: Implementing the Feature

### Step 1: Add Deprecation Warnings to Legacy Classes

**Files to modify**:
- `robot_sf/gym_env/env_config.py` (4 classes: `BaseEnvSettings`, `RobotEnvSettings`, `EnvSettings`, `PedEnvSettings`)

**Note**: Do NOT modify `sim_config.py` - `SimulationSettings` is canonical and used by unified config

**Pattern**:
```python
import warnings
from dataclasses import dataclass, field

@dataclass
class EnvSettings:
    """Legacy config class - use RobotSimulationConfig instead."""
    
    # ... existing fields ...
    
    def __post_init__(self):
        warnings.warn(
            "EnvSettings is deprecated and will be removed in a future version. "
            "Use RobotSimulationConfig from robot_sf.gym_env.unified_config instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # ... existing validation code ...
```

**Apply to**:
- `BaseEnvSettings` → Recommend `BaseSimulationConfig`
- `RobotEnvSettings` → Recommend `RobotSimulationConfig`
- `EnvSettings` → Recommend `RobotSimulationConfig`
- `PedEnvSettings` → Recommend `PedestrianSimulationConfig`

### Step 2: Create Configuration Documentation

**Create**: `docs/architecture/configuration.md`

**Sections to include**:
1. Overview - What is the configuration system?
2. Precedence Hierarchy - Code < YAML < Runtime (with examples)
3. Configuration Modules - Canonical vs legacy module map
4. Unified Config Classes - Description of each class in `unified_config.py`
5. YAML Configuration - How scenario/baseline configs work
6. Runtime Overrides - Factory function parameter examples
7. Migration Guide - Legacy → unified conversions with code examples
8. External Configuration - fast-pysf and other external configs
9. Best Practices - When to use each configuration level

**Reference**: See `research.md` and `data-model.md` for content details

### Step 3: Update Documentation Index

**Modify**: `docs/README.md`

**Add link** in appropriate section (e.g., "Architecture" or "Development"):
```markdown
## Architecture

- [Configuration Hierarchy](architecture/configuration.md) - Configuration precedence, unified vs legacy configs, migration guide
```

**Modify**: `docs/dev_guide.md`

**Add reference** in "Configuration hierarchy" section:
```markdown
### Configuration hierarchy (CRITICAL)
**Always use factory functions** — never instantiate gym environments directly.

For details on configuration precedence and unified config classes, see [Configuration Architecture](./architecture/configuration.md).
```

### Step 4: Create Deprecation Tests

**Create**: `tests/test_gym_env/test_config_deprecation.py`

**Test cases**:
```python
import warnings
import pytest
from robot_sf.gym_env.env_config import (
    BaseEnvSettings,
    RobotEnvSettings,
    EnvSettings,
    PedEnvSettings,
)

def test_base_env_settings_deprecated():
    with pytest.warns(DeprecationWarning, match="BaseSimulationConfig"):
        config = BaseEnvSettings()

def test_robot_env_settings_deprecated():
    with pytest.warns(DeprecationWarning, match="RobotSimulationConfig"):
        config = RobotEnvSettings()

def test_env_settings_deprecated():
    with pytest.warns(DeprecationWarning, match="RobotSimulationConfig"):
        config = EnvSettings()

def test_ped_env_settings_deprecated():
    with pytest.warns(DeprecationWarning, match="PedestrianSimulationConfig"):
        config = PedEnvSettings()

def test_deprecation_message_content():
    """Verify warning message includes required information"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = EnvSettings()
        
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        message = str(w[0].message)
        
        assert "EnvSettings" in message
        assert "deprecated" in message.lower()
        assert "RobotSimulationConfig" in message
        assert "unified_config" in message
```

### Step 5: Run Tests and Verify

**Commands**:
```bash
# Run deprecation tests
uv run pytest tests/test_gym_env/test_config_deprecation.py -v

# Run full test suite (verify no regressions)
uv run pytest tests -v

# Expected: All tests pass, deprecation warnings visible in output
```

**Verify**:
- All new deprecation tests pass
- All existing tests pass (backward compatibility maintained)
- Warnings appear in test output but don't cause failures

### Step 6: Quality Gates

**Run before committing**:
```bash
# Format and lint
uv run ruff check --fix . && uv run ruff format .

# Type check
uvx ty check . --exit-zero

# Full test suite
uv run pytest tests
```

---

## For Users: Quick Migration Scenarios

### Scenario 1: Basic Robot Environment

**Legacy**:
```python
from robot_sf.gym_env.env_config import EnvSettings
config = EnvSettings()
# ... use config ...
```

**Unified**:
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig
config = RobotSimulationConfig()
# ... use config ... (same API)
```

### Scenario 2: Pedestrian Environment

**Legacy**:
```python
from robot_sf.gym_env.env_config import PedEnvSettings
config = PedEnvSettings(
    ego_ped_config=my_ped_config
)
```

**Unified**:
```python
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
config = PedestrianSimulationConfig(
    ego_ped_config=my_ped_config
)
```

### Scenario 3: Custom Simulation Settings

**Legacy**:
```python
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.sim.sim_config import SimulationSettings

config = EnvSettings(
    sim_config=SimulationSettings(
        sim_time_in_secs=100.0,
        difficulty=2
    )
)
```

**Unified**:
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.sim.sim_config import SimulationSettings

config = RobotSimulationConfig(
    sim_config=SimulationSettings(
        sim_time_in_secs=100.0,
        difficulty=2
    )
)
```

**Note**: `SimulationSettings` is still canonical - it's used by both legacy and unified configs. Only the outer config class changes (`EnvSettings` → `RobotSimulationConfig`).

---

## Timeline and Phases

**Phase 1 (This Feature - 244)**:
- ✅ Document configuration hierarchy
- ✅ Add deprecation warnings to legacy classes
- ✅ Create migration guide
- ✅ Verify backward compatibility

**Phase 2 (Future)**:
- Monitor usage of legacy vs unified configs
- Announce removal timeline (e.g., "will be removed in v2.0")
- Update examples and docs to use only unified configs

**Phase 3 (Future)**:
- Remove legacy config classes from codebase
- Update tests to only use unified configs
- Bump major version (breaking change)

---

## Troubleshooting

### Q: I'm getting DeprecationWarnings in my tests

**A**: This is expected! Legacy configs are still functional but deprecated. To migrate:
1. Find the canonical replacement in the warning message
2. Update imports from `env_config` to `unified_config`
3. Change class name (fields remain the same)

### Q: Can I suppress the warnings temporarily?

**A**: Yes, but migration is recommended:
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

### Q: Will my code break when I update?

**A**: No! This feature is non-breaking. Legacy configs continue to work with warnings. You can migrate at your own pace.

### Q: Where do I find the full migration guide?

**A**: `docs/architecture/configuration.md` → "Migration Guide" section

### Q: What if I'm using YAML configs?

**A**: YAML configs map to unified classes automatically. See `docs/architecture/configuration.md` → "YAML Configuration" section.

---

## Success Checklist

**For Feature Completion**:
- [ ] Deprecation warnings added to all 4 legacy classes
- [ ] `docs/architecture/configuration.md` created with all sections
- [ ] `docs/README.md` updated with link to config docs
- [ ] `docs/dev_guide.md` references config docs
- [ ] `tests/test_gym_env/test_config_deprecation.py` created
- [ ] All deprecation tests pass
- [ ] Full test suite passes (no regressions)
- [ ] Quality gates pass (ruff, type check, pytest)

**Verification Commands**:
```bash
# Quick verification
uv run pytest tests/test_gym_env/test_config_deprecation.py -v
uv run pytest tests -v
uv run ruff check .
```
