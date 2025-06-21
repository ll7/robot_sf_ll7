# Migration Guide: Environment Refactoring

> 📚 **Documentation Navigation**: [← Back to Refactoring Index](README.md) | [🚀 Deployment Status](DEPLOYMENT_READY.md) | [📋 Plan](refactoring_plan.md) | [📊 Summary](refactoring_summary.md)

## Overview

This guide provides step-by-step instructions for migrating from the current inconsistent environment abstractions to the new unified hierarchy.

## Current Problems

### Before Refactoring:
- **Inconsistent inheritance**: Some environments extend `BaseEnv`, others extend `gymnasium.Env` directly
- **Code duplication**: Recording, visualization, and initialization logic repeated across environments
- **Configuration chaos**: Multiple overlapping config classes (`EnvSettings`, `RobotEnvSettings`, `PedEnvSettings`)
- **Unclear responsibilities**: Environment classes have overlapping functionality

### After Refactoring:
- **Consistent hierarchy**: All environments follow the same abstract base class pattern
- **Shared functionality**: Common code in base classes eliminates duplication
- **Unified configuration**: Clear, hierarchical configuration classes
- **Clear separation of concerns**: Each class has specific, well-defined responsibilities

## Migration Steps

### Phase 1: Update Imports and Factory Usage

#### Old Way:
```python
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.gym_env.env_config import EnvSettings, PedEnvSettings

# Creating environments manually
robot_env = RobotEnv(env_config=EnvSettings())
ped_env = PedestrianEnv(env_config=PedEnvSettings(), robot_model=model)
```

#### New Way:
```python
from robot_sf.gym_env.environment_factory import EnvironmentFactory
from robot_sf.gym_env.unified_config import RobotSimulationConfig, PedestrianSimulationConfig

# Using factory for consistent interface
robot_env = EnvironmentFactory.create_robot_env(
    config=RobotSimulationConfig(),
    debug=True
)

ped_env = EnvironmentFactory.create_pedestrian_env(
    config=PedestrianSimulationConfig(),
    robot_model=model,
    debug=True
)

# Or use convenience functions
from robot_sf.gym_env.environment_factory import make_robot_env, make_pedestrian_env

robot_env = make_robot_env(debug=True)
ped_env = make_pedestrian_env(robot_model=model, debug=True)
```

### Phase 2: Update Configuration Classes

#### Old Configuration:
```python
from robot_sf.gym_env.env_config import EnvSettings, RobotEnvSettings, PedEnvSettings

# Multiple similar config classes
robot_config = RobotEnvSettings(use_image_obs=True)
ped_config = PedEnvSettings(ego_ped_config=UnicycleDriveSettings())
```

#### New Configuration:
```python
from robot_sf.gym_env.unified_config import (
    RobotSimulationConfig, 
    ImageRobotConfig, 
    PedestrianSimulationConfig
)

# Clear, hierarchical configuration
robot_config = RobotSimulationConfig()
image_robot_config = ImageRobotConfig()  # Automatically enables image obs
ped_config = PedestrianSimulationConfig()
```

### Phase 3: Migrate Custom Environments

#### Old Custom Environment:
```python
from gymnasium import Env
from robot_sf.gym_env.env_config import EnvSettings

class CustomRobotEnv(Env):
    def __init__(self, env_config=EnvSettings(), debug=False):
        # Duplicate initialization code
        self.env_config = env_config
        self.debug = debug
        # ... lots of boilerplate
        
    def render(self):
        # Duplicate rendering logic
        pass
        
    def exit(self):
        # Duplicate cleanup logic  
        pass
```

#### New Custom Environment:
```python
from robot_sf.gym_env.abstract_envs import SingleAgentEnv
from robot_sf.gym_env.unified_config import RobotSimulationConfig

class CustomRobotEnv(SingleAgentEnv):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = RobotSimulationConfig()
        super().__init__(config=config, **kwargs)
        
    def _setup_environment(self):
        # Only implement environment-specific setup
        self.map_def = self.config.map_pool.choose_random_map()
        # ... custom logic
        
    def _create_spaces(self):
        # Only implement space creation
        return action_space, observation_space
```

### Phase 4: Update Test Files

#### Old Tests:
```python
def test_env_creation():
    env = RobotEnv()
    assert env is not None
    
def test_ped_env():
    env = PedestrianEnv(robot_model=model)
    assert env is not None
```

#### New Tests:
```python
def test_env_creation():
    env = make_robot_env()
    assert env is not None
    
def test_ped_env():
    env = make_pedestrian_env(robot_model=model)
    assert env is not None
    
def test_factory_consistency():
    # All environments follow same interface
    envs = [
        make_robot_env(),
        make_image_robot_env(),
        make_pedestrian_env(robot_model=model),
    ]
    for env in envs:
        assert hasattr(env, 'step')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'render')
        assert hasattr(env, 'exit')
```

## File-by-File Migration Plan

### 1. Core Files (Create New)
- ✅ `robot_sf/gym_env/abstract_envs.py` - Abstract base classes
- ✅ `robot_sf/gym_env/unified_config.py` - Consolidated configuration
- ✅ `robot_sf/gym_env/environment_factory.py` - Factory pattern implementation

### 2. Environment Files (Refactor)
- 🔄 `robot_sf/gym_env/pedestrian_env.py` - Extend SingleAgentEnv
- 🔄 `robot_sf/gym_env/robot_env.py` - Update to use unified config
- 🔄 `robot_sf/gym_env/empty_robot_env.py` - Merge with RobotEnv or refactor
- 🔄 `robot_sf/gym_env/multi_robot_env.py` - Extend MultiAgentEnv
- 🔄 `robot_sf/gym_env/robot_env_with_image.py` - Potentially merge with RobotEnv

### 3. Configuration Files (Update)
- 🔄 `robot_sf/gym_env/env_config.py` - Add backward compatibility imports

### 4. Example Files (Update)
- 🔄 `examples/demo_pedestrian.py` - Use factory pattern
- 🔄 `examples/demo_defensive.py` - Use factory pattern
- 🔄 All example files - Update to new interface

### 5. Test Files (Update)
- 🔄 `tests/env_test.py` - Use factory pattern
- 🔄 `tests/test_robot_env_with_image_integration.py` - Update imports
- 🔄 All test files - Update to new interface

## Backward Compatibility Strategy

### Phase 1: Parallel Implementation
- Keep existing classes working
- Add new classes alongside old ones
- Provide deprecation warnings

```python
# In env_config.py
import warnings
from robot_sf.gym_env.unified_config import RobotSimulationConfig

@dataclass  
class EnvSettings(RobotSimulationConfig):
    """Deprecated: Use RobotSimulationConfig instead."""
    def __post_init__(self):
        warnings.warn(
            "EnvSettings is deprecated. Use RobotSimulationConfig instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__post_init__()
```

### Phase 2: Migration Period
- Update examples to use new pattern
- Update documentation
- Encourage users to migrate

### Phase 3: Cleanup
- Remove deprecated classes
- Clean up imports
- Finalize new architecture

## Benefits of Migration

### 1. Code Quality
- **Reduced duplication**: 50% less code in environment classes
- **Consistent interface**: All environments follow same patterns
- **Better maintainability**: Changes only need to be made once

### 2. Developer Experience
- **Easier testing**: Consistent interfaces make testing simpler
- **Clearer documentation**: One set of patterns to learn
- **Better IDE support**: Consistent typing and interfaces

### 3. Extensibility
- **Easy to add new environments**: Just extend appropriate base class
- **Flexible configuration**: Hierarchical config system
- **Plugin-friendly**: Factory pattern supports easy environment registration

## Testing Strategy

### 1. Backward Compatibility Tests
```python
def test_backward_compatibility():
    # Old interface should still work
    from robot_sf.gym_env.env_config import EnvSettings
    from robot_sf.gym_env.robot_env import RobotEnv
    
    env = RobotEnv(env_config=EnvSettings())
    assert env is not None
```

### 2. New Interface Tests
```python
def test_new_interface():
    # New interface should work better
    env = make_robot_env()
    assert env is not None
    assert hasattr(env, 'config')
    assert isinstance(env.config, RobotSimulationConfig)
```

### 3. Migration Tests
```python
def test_equivalent_behavior():
    # Old and new should behave the same
    old_env = RobotEnv()
    new_env = make_robot_env()
    
    old_obs, _ = old_env.reset()
    new_obs, _ = new_env.reset()
    
    # Should have same structure
    assert old_obs.keys() == new_obs.keys()
```

This migration preserves all existing functionality while providing a much cleaner, more maintainable architecture.
