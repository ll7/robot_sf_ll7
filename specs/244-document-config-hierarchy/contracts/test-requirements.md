# Test Contracts: Configuration Hierarchy Documentation

**Feature**: 244-document-config-hierarchy  
**Date**: 2025-11-11

## Test Requirements Overview

This feature requires:
1. Deprecation warning emission tests
2. Backward compatibility verification (existing tests pass)
3. Documentation discoverability tests
4. Migration example validation

## Contract 1: Deprecation Warnings

### Test: Legacy Config Classes Emit DeprecationWarning

**Given**: A user instantiates a legacy config class  
**When**: The class `__post_init__` method runs  
**Then**: A `DeprecationWarning` is emitted with the canonical replacement class name

**Test Cases**:

```python
def test_env_settings_deprecated():
    """EnvSettings emits DeprecationWarning mentioning RobotSimulationConfig"""
    with pytest.warns(DeprecationWarning, match="RobotSimulationConfig"):
        from robot_sf.gym_env.env_config import EnvSettings
        config = EnvSettings()
    
def test_ped_env_settings_deprecated():
    """PedEnvSettings emits DeprecationWarning mentioning PedestrianSimulationConfig"""
    with pytest.warns(DeprecationWarning, match="PedestrianSimulationConfig"):
        from robot_sf.gym_env.env_config import PedEnvSettings
        config = PedEnvSettings()

def test_robot_env_settings_deprecated():
    """RobotEnvSettings emits DeprecationWarning mentioning RobotSimulationConfig"""
    with pytest.warns(DeprecationWarning, match="RobotSimulationConfig"):
        from robot_sf.gym_env.env_config import RobotEnvSettings
        config = RobotEnvSettings()

def test_base_env_settings_deprecated():
    """BaseEnvSettings emits DeprecationWarning mentioning BaseSimulationConfig"""
    with pytest.warns(DeprecationWarning, match="BaseSimulationConfig"):
        from robot_sf.gym_env.env_config import BaseEnvSettings
        config = BaseEnvSettings()
```

**Acceptance Criteria**:
- [ ] All four legacy classes emit `DeprecationWarning` on instantiation
- [ ] Warning message includes the canonical replacement class name
- [ ] Warning `stacklevel=2` points to user code location
- [ ] Warning does not prevent successful instantiation

**Test File**: `tests/test_gym_env/test_config_deprecation.py`

---

## Contract 2: Backward Compatibility

### Test: Existing Tests Pass After Deprecation Warnings

**Given**: Deprecation warnings added to legacy config classes  
**When**: The full test suite runs  
**Then**: All existing tests pass (no regressions)

**Verification Command**:
```bash
uv run pytest tests -v
```

**Acceptance Criteria**:
- [ ] Zero test failures introduced by deprecation warnings
- [ ] Zero test errors introduced by deprecation warnings
- [ ] Warnings may appear in test output but don't break functionality
- [ ] Test count remains stable (no skipped tests)

**Test Scope**: All tests in `tests/` directory

---

## Contract 3: Documentation Discoverability

### Test: Configuration Documentation is Linked from Index

**Given**: The new `docs/architecture/configuration.md` file exists  
**When**: A user navigates to `docs/README.md`  
**Then**: They find a link to the configuration documentation

**Manual Verification**:
1. Open `docs/README.md`
2. Search for "configuration" or "config"
3. Verify link to `architecture/configuration.md` exists
4. Click link and verify it loads the correct file

**Acceptance Criteria**:
- [ ] `docs/README.md` contains link to `architecture/configuration.md`
- [ ] `docs/dev_guide.md` references configuration documentation
- [ ] Link text is descriptive (e.g., "Configuration Hierarchy and Best Practices")
- [ ] File path is correct (relative or absolute as appropriate)

**Test Type**: Manual verification or documentation linter check

---

## Contract 4: Migration Examples Are Valid

### Test: Migration Code Examples Are Syntactically Correct

**Given**: Migration examples in `docs/architecture/configuration.md`  
**When**: Examples are extracted and run through Python syntax checker  
**Then**: All examples parse without syntax errors

**Verification Approach**:
```python
import ast

def test_migration_examples_valid_syntax():
    """Extract and validate all code examples from configuration.md"""
    doc_path = "docs/architecture/configuration.md"
    with open(doc_path) as f:
        content = f.read()
    
    # Extract code blocks (simplified - actual implementation would be more robust)
    code_blocks = extract_python_code_blocks(content)
    
    for i, code in enumerate(code_blocks):
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Code block {i} has syntax error: {e}")
```

**Acceptance Criteria**:
- [ ] All Python code examples in migration guide are syntactically valid
- [ ] Import statements reference actual modules
- [ ] Class names match actual config classes
- [ ] Examples demonstrate working conversions (could be run in a test environment)

**Test File**: `tests/test_documentation/test_config_examples.py` (optional, nice-to-have)

---

## Contract 5: Configuration Precedence Behavior

### Test: Runtime Parameters Override YAML and Code Defaults

**Given**: A config with code default, YAML override, and runtime override  
**When**: The environment is created via factory function  
**Then**: Runtime parameter takes precedence

**Example Test**:
```python
def test_config_precedence_runtime_wins():
    """Runtime parameters override YAML and code defaults"""
    # Code default: sim_time_in_secs = 200.0 (in SimulationSettings)
    # YAML override: sim_time_in_secs = 150.0 (hypothetical)
    # Runtime override: sim_time_in_secs = 100.0 (via config kwarg)
    
    from robot_sf.gym_env.unified_config import RobotSimulationConfig
    from robot_sf.sim.sim_config import SimulationSettings
    
    runtime_config = RobotSimulationConfig(
        sim_config=SimulationSettings(sim_time_in_secs=100.0)
    )
    
    env = make_robot_env(config=runtime_config)
    
    # Verify runtime value is used
    assert env.config.sim_config.sim_time_in_secs == 100.0
```

**Acceptance Criteria**:
- [ ] Runtime config overrides code defaults
- [ ] Runtime config overrides YAML-loaded configs (when both present)
- [ ] Precedence behavior matches documentation
- [ ] Test covers at least one parameter from each config level

**Test File**: `tests/test_gym_env/test_config_precedence.py` (optional, nice-to-have)

---

## Contract 6: Deprecation Warning Message Content

### Test: Warning Messages Include Required Information

**Given**: A legacy config class is instantiated  
**When**: The deprecation warning is emitted  
**Then**: The message contains:
- Legacy class name
- Canonical replacement class name
- Module path for replacement
- "deprecated" keyword

**Example Test**:
```python
def test_deprecation_message_format():
    """Deprecation warning includes all required information"""
    import warnings
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from robot_sf.gym_env.env_config import EnvSettings
        config = EnvSettings()
        
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        message = str(w[0].message)
        
        # Check required content
        assert "EnvSettings" in message
        assert "deprecated" in message.lower()
        assert "RobotSimulationConfig" in message
        assert "robot_sf.gym_env.unified_config" in message
```

**Acceptance Criteria**:
- [ ] Message includes legacy class name
- [ ] Message includes canonical replacement class name
- [ ] Message includes full import path for replacement
- [ ] Message is user-friendly and actionable

**Test File**: `tests/test_gym_env/test_config_deprecation.py`

---

## Test Execution Plan

### Phase 1: Core Deprecation Tests
1. Create `tests/test_gym_env/test_config_deprecation.py`
2. Implement warning emission tests (Contract 1)
3. Implement warning message format tests (Contract 6)
4. Run tests: `uv run pytest tests/test_gym_env/test_config_deprecation.py -v`

### Phase 2: Backward Compatibility Verification
1. Run full test suite: `uv run pytest tests -v`
2. Verify zero regressions (Contract 2)
3. Document any expected warning counts in test output

### Phase 3: Documentation Validation (Manual)
1. Verify `docs/README.md` link (Contract 3)
2. Verify `docs/dev_guide.md` references (Contract 3)
3. Review migration examples for syntax (Contract 4)

### Phase 4: Optional Extended Tests
1. Implement config precedence tests (Contract 5) - nice-to-have
2. Implement migration example extractor/validator (Contract 4) - nice-to-have

---

## Success Criteria Summary

All contracts must pass for feature completion:
- ✅ Contract 1: Deprecation warnings emit correctly
- ✅ Contract 2: No test regressions (all existing tests pass)
- ✅ Contract 3: Documentation is discoverable
- ✅ Contract 4: Migration examples are valid
- ⚠️ Contract 5: Config precedence behavior (optional)
- ✅ Contract 6: Warning messages are complete

Minimum viable: Contracts 1, 2, 3, 4, 6 must pass.  
Optional enhancement: Contract 5 adds value but not required for initial release.
