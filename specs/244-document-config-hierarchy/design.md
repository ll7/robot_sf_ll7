# Design: Configuration Hierarchy Documentation

**Feature**: Issue #244  
**Author**: GitHub Copilot  
**Date**: 2025-01-11  
**Status**: Implementation Complete

## Purpose

Document the existing configuration hierarchy in robot_sf and add deprecation warnings to legacy config classes, preparing for future consolidation without breaking existing code.

## Problem Statement

The robot_sf project has evolved multiple configuration systems over time, leading to:

1. **Unclear precedence**: Developers don't know whether code defaults, YAML files, or runtime parameters take priority
2. **Multiple config modules**: `unified_config.py`, `env_config.py`, `sim_config.py`, `map_config.py` - which should be used?
3. **Legacy patterns persist**: New code sometimes uses outdated config classes
4. **No migration path**: Users don't know how to modernize their code

This creates confusion, maintenance burden, and technical debt accumulation.

## Goals

### In Scope (This Issue - Phases 1 & 2)

✅ Document configuration precedence hierarchy  
✅ Identify canonical vs legacy config modules  
✅ Add deprecation warnings to legacy classes  
✅ Create migration guide with examples  
✅ Make configuration docs discoverable  

### Out of Scope (Deferred to Phase 3 / v3.0)

❌ Consolidate config modules into single package  
❌ Remove legacy config classes (breaking change)  
❌ Refactor existing code to use unified configs  
❌ Add schema validation for YAML configs  

**Rationale for deferral**: Phase 3 requires breaking changes and extensive refactoring, better suited for a major version bump with dedicated planning.

## Architecture

### Current Configuration Hierarchy (Precedence: Low → High)

```
┌─────────────────────────────────────────┐
│  1. Code Defaults (Lowest Priority)    │
│     - unified_config.py classes         │
│     - BaseSimulationConfig              │
│     - RobotSimulationConfig             │
│     - ImageRobotConfig                  │
│     - PedestrianSimulationConfig        │
└─────────────────────────────────────────┘
                  ↓ (overridden by)
┌─────────────────────────────────────────┐
│  2. YAML Files (Medium Priority)        │
│     - configs/scenarios/*.yaml          │
│     - configs/baselines/*.yaml          │
│     - Loaded and merged at runtime      │
└─────────────────────────────────────────┘
                  ↓ (overridden by)
┌─────────────────────────────────────────┐
│  3. Runtime Parameters (Highest)        │
│     - Factory function kwargs           │
│     - Environment-specific overrides    │
│     - Direct config object mutations    │
└─────────────────────────────────────────┘
```

### Config Module Status

| Module | Status | Replacement | Notes |
|--------|--------|-------------|-------|
| `unified_config.py` | ✅ **Canonical** | N/A | Use for all new code |
| `sim_config.py` | ✅ **Canonical** | N/A | `SimulationSettings` used by unified config |
| `map_config.py` | ✅ **Canonical** | N/A | Map definitions, still current |
| `env_config.py` | ⚠️ **Legacy** | `unified_config.py` | To be deprecated |
| `fast-pysf/config.py` | 🔗 **External** | N/A | Managed by subtree, pass-through |

### Deprecation Strategy

**Approach**: Soft deprecation with warnings (non-breaking)

```python
# In env_config.py
import warnings

class EnvSettings:
    """DEPRECATED: Use unified_config.RobotSimulationConfig instead.
    
    This class will be removed in v3.0.
    See docs/architecture/configuration.md for migration guide.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "EnvSettings is deprecated. Use RobotSimulationConfig from "
            "robot_sf.gym_env.unified_config instead. "
            "See docs/architecture/configuration.md for migration guide.",
            DeprecationWarning,
            stacklevel=2  # Show caller location
        )
        # Original implementation continues...
```

**Benefits**:
- ✅ Non-breaking: existing code continues to work
- ✅ Visible: warnings guide developers to modern patterns
- ✅ Traceable: `stacklevel=2` shows where legacy config is used
- ✅ Documented: clear migration path provided

**Drawbacks**:
- ⚠️ Still maintains two systems until v3.0
- ⚠️ Test output may become noisy (can be filtered)

## Implementation Plan

### Documentation Structure

**New file**: `docs/architecture/configuration.md`

**Content outline**:
1. **Overview**: Why configuration matters, what this doc covers
2. **Precedence Rules**: Code < YAML < Runtime (with examples)
3. **Module Guide**: Which configs to use (canonical vs legacy table)
4. **YAML Integration**: How YAML files map to unified config classes
5. **Migration Guide**: Legacy → Unified conversion examples
6. **External Configs**: Relationship with fast-pysf
7. **Future Plans**: Link to Phase 3 tracking issue (when created)

### Deprecation Implementation

**Files to modify**:
- `robot_sf/gym_env/env_config.py` - Add warnings to `BaseEnvSettings`, `RobotEnvSettings`, `EnvSettings`, `PedEnvSettings`

**Note**: `sim_config.py` is NOT modified - `SimulationSettings` remains canonical (used by unified config)

**Warning message template**:
```
"{ClassName} is deprecated. Use {ReplacementClass} from robot_sf.gym_env.unified_config instead. See docs/architecture/configuration.md for migration guide."
```

**Test considerations**:
- Warnings should **not** fail tests (use `warnings.warn()`, not exceptions)
- Test fixtures may need `warnings.filterwarnings("ignore", category=DeprecationWarning)` if they use legacy configs intentionally
- Add new tests validating that warnings are emitted correctly

### Integration Points

**Links to add**:
- `docs/README.md` → Add architecture/configuration.md under "Architecture" section
- `docs/dev_guide.md` → Update "Configuration hierarchy" section, add link in "Quick links"
- Legacy config class docstrings → Reference migration guide
- `CHANGELOG.md` → Document deprecations and new docs

## Testing Strategy

### Documentation Tests
- [ ] Developer can find configuration precedence in < 2 minutes
- [ ] Migration examples are copy-paste ready and work

### Functional Tests
- [ ] Deprecation warnings appear when legacy configs instantiated
- [ ] All existing tests pass (non-breaking change)
- [ ] New tests verify configuration precedence behavior

### Validation Tests
```python
def test_config_precedence_code_default():
    """Verify code defaults are used when no overrides present."""
    config = RobotSimulationConfig()
    assert config.some_param == DEFAULT_VALUE

def test_config_precedence_yaml_override():
    """Verify YAML overrides code defaults."""
    config = load_config_from_yaml("test.yaml")
    assert config.some_param == YAML_VALUE

def test_config_precedence_runtime_override():
    """Verify runtime kwargs override YAML and code defaults."""
    config = RobotSimulationConfig(some_param=RUNTIME_VALUE)
    assert config.some_param == RUNTIME_VALUE

def test_deprecation_warning_env_settings():
    """Verify EnvSettings emits deprecation warning."""
    with pytest.warns(DeprecationWarning, match="EnvSettings is deprecated"):
        settings = EnvSettings()
```

## Migration Examples

### Example 1: EnvSettings → RobotSimulationConfig

**Before (Legacy)**:
```python
from robot_sf.gym_env.env_config import EnvSettings

env_settings = EnvSettings(
    obstacles=map_def.obstacles,
    robot_radius=0.5,
)
```

**After (Unified)**:
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig

config = RobotSimulationConfig()
config.obstacle_map = map_def  # Map handling integrated
config.robot_radius = 0.5
```

## Trade-offs and Decisions

### Decision 1: Soft Deprecation vs Immediate Removal

**Chosen**: Soft deprecation with warnings

**Alternatives considered**:
1. ❌ Immediate removal - Too risky, breaks all existing code
2. ❌ No deprecation - Doesn't guide users, perpetuates legacy usage
3. ✅ Soft deprecation - Balances compatibility with guidance

**Rationale**: Non-breaking approach allows gradual migration, protects existing users, while still guiding new development toward modern patterns.

### Decision 2: Documentation Location

**Chosen**: `docs/architecture/configuration.md`

**Alternatives considered**:
1. ❌ In dev_guide.md - Too long, would clutter main guide
2. ❌ In README.md - Wrong audience (users vs developers)
3. ✅ Dedicated architecture doc - Clear separation of concerns

**Rationale**: Architecture docs are for understanding system structure; dev_guide is for workflows. Separation keeps both focused.

### Decision 3: Phase 3 Deferral

**Chosen**: Defer consolidation to v3.0

**Rationale**:
- Phase 1-2 delivers immediate value without risk
- Consolidation requires breaking changes better suited for major version
- Time investment (20+ hours) justifies dedicated planning
- Current approach establishes migration path first, simplifying future consolidation

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Documentation becomes outdated | Medium | Medium | Link from code comments, include in PR review checklist |
| Test output too noisy with warnings | High | Low | Add `pytest.ini` filter config for known deprecations |
| Users ignore deprecation warnings | Medium | Low | Clear migration guide, examples in docs, changelog entry |
| Precedence rules don't match reality | Low | High | Add tests validating documented behavior, verify in PR |
| Legacy configs have subtle behavioral differences | Low | High | Test migration examples, document gotchas in guide |

## Success Metrics

**Immediate (PR merge)**:
- ✅ Configuration docs exist and are discoverable
- ✅ All tests pass with deprecation warnings
- ✅ Deprecation warnings appear in test output

**Short-term (1-2 months)**:
- ✅ Zero new usage of legacy configs in merged PRs
- ✅ Users successfully migrate following guide (GitHub issues reference it)

**Long-term (v3.0 timeline)**:
- ✅ Phase 3 consolidation uses this docs as baseline
- ✅ Removal of legacy configs is smooth (users already migrated)

## Future Work (Phase 3 - v3.0)

When we're ready for breaking changes:

1. **Consolidate modules** into `robot_sf/config/` package
2. **Remove legacy classes** (EnvSettings, PedEnvSettings) - Note: SimulationSettings is canonical and remains
3. **Add schema validation** for YAML configs (e.g., using Pydantic)
4. **Unify YAML and code config formats** for consistency
5. **Create config versioning system** for backward compatibility

**Tracking**: Create separate issue for Phase 3 with link from this documentation.

## References

- Issue #244: [Document configuration hierarchy and deprecate legacy config classes](https://github.com/ll7/robot_sf_ll7/issues/244)
- Related docs: `docs/dev_guide.md`, `docs/refactoring/`
- Python deprecation warnings: https://docs.python.org/3/library/warnings.html
- Semantic versioning: https://semver.org/
