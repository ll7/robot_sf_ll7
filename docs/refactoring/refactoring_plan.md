# Environment Refactoring Plan

> 📚 **Documentation Navigation**: [← Back to Refactoring Index](README.md) | [🚀 Deployment Status](DEPLOYMENT_READY.md) | [🔄 Migration Guide](migration_guide.md) | [📊 Summary](refactoring_summary.md)

## Goal
Create a consistent, extensible environment hierarchy that eliminates code duplication and provides clear separation of concerns.

## Proposed Hierarchy

```
gymnasium.Env
├── BaseSimulationEnv (abstract base)
│   ├── SingleAgentEnv (abstract)
│   │   ├── RobotEnv
│   │   ├── RobotEnvWithImage  
│   │   └── PedestrianEnv
│   └── MultiAgentEnv (abstract)
│       └── MultiRobotEnv
└── SimpleRobotEnv (lightweight, separate)
```

## Phase 1: Consolidate Configuration Classes

### Current Issues:
- `EnvSettings` - Basic configuration
- `RobotEnvSettings` - Extends BaseEnvSettings, adds image config
- `PedEnvSettings` - Extends EnvSettings, adds ego pedestrian config

### Proposed Solution:
```python
@dataclass
class BaseSimulationConfig:
    """Core simulation configuration shared by all environments"""
    sim_config: SimulationSettings
    map_pool: MapDefinitionPool
    lidar_config: LidarScannerSettings
    
@dataclass  
class RobotConfig(BaseSimulationConfig):
    """Robot-specific configuration"""
    robot_config: Union[DifferentialDriveSettings, BicycleDriveSettings]
    
@dataclass
class ImageRobotConfig(RobotConfig):
    """Robot configuration with image observations"""
    image_config: ImageSensorSettings 
    use_image_obs: bool = True
    
@dataclass
class PedestrianConfig(RobotConfig):
    """Configuration for pedestrian environments"""
    ego_ped_config: UnicycleDriveSettings
```

## Phase 2: Create Abstract Base Classes

### BaseSimulationEnv
- Common initialization (simulator, map, recording)
- Shared utility methods (exit, save_recording, render setup)
- Abstract methods for environment-specific logic

### SingleAgentEnv  
- Single robot/agent simulation logic
- Common step/reset patterns
- Sensor and collision detection setup

### MultiAgentEnv
- Multi-agent coordination
- Vectorized operations
- Parallel simulation management

## Phase 3: Refactor Existing Environments

### 1. Update PedestrianEnv
- Extend SingleAgentEnv instead of gymnasium.Env
- Remove duplicated BaseEnv functionality
- Use consolidated configuration classes

### 2. Update EmptyRobotEnv  
- Extend SingleAgentEnv
- Remove duplicated initialization code
- Consider merging with RobotEnv as a configuration option

### 3. Consolidate RobotEnv variants
- Make image observations a configuration option in RobotEnv
- Consider obstacle forces as a configuration flag
- Reduce the number of environment classes

### 4. Improve MultiRobotEnv
- Ensure consistent interface with single-agent environments
- Share common patterns with SingleAgentEnv where possible

## Phase 4: Factory Pattern Implementation

```python
class EnvironmentFactory:
    @staticmethod
    def create_robot_env(config: RobotConfig, **kwargs) -> SingleAgentEnv:
        if config.use_image_obs:
            return RobotEnvWithImage(config, **kwargs)
        return RobotEnv(config, **kwargs)
    
    @staticmethod  
    def create_pedestrian_env(config: PedestrianConfig, **kwargs) -> SingleAgentEnv:
        return PedestrianEnv(config, **kwargs)
        
    @staticmethod
    def create_multi_robot_env(config: RobotConfig, num_robots: int, **kwargs) -> MultiAgentEnv:
        return MultiRobotEnv(config, num_robots, **kwargs)
```

## Benefits

1. **Consistent Interface**: All environments follow the same patterns
2. **Reduced Duplication**: Shared functionality in base classes
3. **Better Maintainability**: Changes to common features only need to be made once
4. **Clearer Responsibilities**: Each class has a specific, well-defined purpose
5. **Easier Testing**: Consistent interfaces make testing simpler
6. **Extensibility**: New environment types can easily extend the hierarchy

## Migration Strategy

1. **Backward Compatibility**: Keep old classes as deprecated wrappers initially
2. **Gradual Migration**: Update one environment at a time
3. **Test Coverage**: Ensure all existing functionality is preserved
4. **Documentation**: Update examples and documentation to use new patterns

## Files to Modify

1. `robot_sf/gym_env/base_env.py` - Expand to BaseSimulationEnv
2. `robot_sf/gym_env/env_config.py` - Consolidate configuration classes  
3. `robot_sf/gym_env/pedestrian_env.py` - Refactor to extend new base
4. `robot_sf/gym_env/empty_robot_env.py` - Refactor or merge
5. `robot_sf/gym_env/robot_env.py` - Update to use consolidated config
6. `robot_sf/gym_env/robot_env_with_image.py` - Potentially merge with RobotEnv
7. `robot_sf/gym_env/multi_robot_env.py` - Update to follow patterns
8. Add: `robot_sf/gym_env/environment_factory.py` - New factory class
9. Add: `robot_sf/gym_env/abstract_envs.py` - Abstract base classes

This refactoring will create a much cleaner, more maintainable codebase with clear responsibilities and reduced duplication.
