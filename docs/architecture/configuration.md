# Configuration Hierarchy and Best Practices

**Status**: Active  
**Last Updated**: 2025-01-11  
**Related**: [Development Guide](../dev_guide.md), [Environment Overview](../ENVIRONMENT.md)

## Overview

The robot_sf project uses a **unified configuration system** to manage simulation parameters, environment settings, and robot behaviors. This document explains the configuration hierarchy, precedence rules, module structure, and migration paths from legacy configuration classes.

**Key Principles**:
- **Type-safe configuration**: All configs are Python dataclasses with explicit types
- **Factory-based creation**: Environments are created via factory functions that accept config objects
- **Clear precedence**: Configuration values follow a well-defined override hierarchy
- **Backward compatible deprecation**: Legacy configs continue to work with warnings

**Purpose**: This configuration system enables:
- Reproducible experiments through versioned config files
- Easy parameter tuning without code changes
- Clear separation of concerns (simulation vs robot vs sensor configs)
- Gradual migration from legacy to unified config classes

---

## Precedence Hierarchy

Configuration values are resolved using a **three-tier precedence hierarchy**:

```
Code Defaults  <  YAML Files  <  Runtime Parameters
   (lowest)                          (highest)
```

### 1. Code Defaults (Lowest Precedence)

Defined in dataclass `field(default=...)` or `field(default_factory=...)` in `robot_sf/gym_env/unified_config.py`.

**Example**:
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig

# All fields use code defaults
config = RobotSimulationConfig()
# sim_config.sim_time_in_secs = 200.0 (default)
# peds_have_static_obstacle_forces = True (default)
# peds_have_obstacle_forces = True (deprecated alias)
```

**When to use**: For sensible defaults that work in most scenarios. These values should represent the "standard" configuration.

### 2. YAML Files (Middle Precedence)

Scenario and baseline configuration files in `configs/scenarios/` and `configs/baselines/` can override code defaults.

**Example YAML** (`configs/scenarios/my_scenario.yaml`):
```yaml
sim_config:
  sim_time_in_secs: 150.0
  difficulty: 2
robot_config:
  max_lin_vel: 1.5
```

**When loaded**: YAML values override code defaults for specified fields only. Unspecified fields retain code defaults.

**When to use**: For reusable experiment configurations, benchmark scenarios, or project-specific presets.

### 3. Runtime Parameters (Highest Precedence)

Parameters passed to factory functions (`make_robot_env()`, `make_image_robot_env()`, etc.) via the `config=` argument or other kwargs.

**Example**:
```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.sim.sim_config import SimulationSettings

# Runtime config overrides everything
runtime_config = RobotSimulationConfig(
    sim_config=SimulationSettings(sim_time_in_secs=100.0),
    peds_have_static_obstacle_forces=True,
    peds_have_robot_repulsion=True,
)

env = make_robot_env(config=runtime_config)
# sim_time_in_secs = 100.0 (runtime override)
# peds_have_static_obstacle_forces = True (runtime override)
# peds_have_robot_repulsion = True (runtime override)
```

**When to use**: For one-off experiments, debugging, parameter sweeps, or ad-hoc overrides during development.

### Precedence Example: All Three Levels

```python
# 1. Code default: sim_time_in_secs = 200.0
# 2. YAML override: sim_time_in_secs = 150.0 (from loaded scenario file)
# 3. Runtime override: sim_time_in_secs = 100.0 (passed to factory)

config = RobotSimulationConfig(
    sim_config=SimulationSettings(sim_time_in_secs=100.0)  # Runtime wins
)
env = make_robot_env(config=config)
assert env.config.sim_config.sim_time_in_secs == 100.0  # Runtime value used
```

**Key Insight**: The precedence hierarchy ensures that code defaults provide a fallback, YAML files enable reusable configurations, and runtime parameters allow maximum flexibility without modifying files.

---

## Configuration Sources

### Code Defaults

**Location**: `robot_sf/gym_env/unified_config.py`

**Characteristics**:
- Immutable within the codebase (require code changes to modify)
- Version-controlled as part of the repository
- Provide sensible starting points for all configuration fields

**Example fields** (from `RobotSimulationConfig`):
```python
@dataclass
class RobotSimulationConfig(BaseSimulationConfig):
    robot_config: DifferentialDriveSettings | BicycleDriveSettings = field(
        default_factory=DifferentialDriveSettings
    )
    use_image_obs: bool = field(default=False)
    peds_have_static_obstacle_forces: bool = field(default=True)
    peds_have_robot_repulsion: bool | None = field(default=None)
    peds_have_obstacle_forces: bool | None = field(default=None)  # deprecated alias
```

### YAML Configuration Files

**Locations**:
- `configs/scenarios/` - Scenario-specific configurations
- `configs/baselines/` - Baseline planner configurations

**Characteristics**:
- Version-controlled for reproducibility
- Human-readable and easy to diff
- Can be shared across experiments
- Support hierarchical structure (nested configs)

**Example YAML structure**:
```yaml
# configs/scenarios/crowded_corridor.yaml
sim_config:
  sim_time_in_secs: 200.0
  difficulty: 3
  peds_speed_mult: 1.3
map_pool:
  map_name: "corridor"
robot_config:
  type: "differential_drive"
  max_lin_vel: 1.2
  max_ang_vel: 1.0
```

**Loading YAML configs**:
YAML files are typically loaded by benchmark runners or training scripts and merged with runtime parameters before being passed to factory functions.

**Scenario manifests (includes)**:
Scenario configuration files can also act as manifests that include other scenario files.
This enables a mix of per-scenario and per-archetype files while keeping a single entry
point for training/benchmark runs.

```yaml
# configs/scenarios/sets/classic_crossing_subset.yaml
includes:
  - ../single/classic_crossing_low.yaml
  - ../single/classic_crossing_medium.yaml
  - ../archetypes/classic_crossing_high.yaml
```

Each included file can contain one or many scenarios. The loader expands includes
relative to the manifest file and preserves ordering.

### Runtime Parameters

**Location**: Passed to factory functions in scripts or interactive sessions

**Characteristics**:
- Ephemeral (not persisted unless explicitly saved)
- Maximum flexibility for experimentation
- Can override any field from code defaults or YAML

**Common patterns**:
```python
# Pattern 1: Override specific fields
config = RobotSimulationConfig()
config.peds_have_static_obstacle_forces = True  # Enable static obstacle forces
config.peds_have_robot_repulsion = True  # Enable pedestrian-robot repulsion
env = make_robot_env(config=config)

# Pattern 2: Create from scratch
config = RobotSimulationConfig(
    sim_config=SimulationSettings(difficulty=4),
    robot_config=DifferentialDriveSettings(max_lin_vel=2.0)
)
env = make_robot_env(config=config)

# Pattern 3: Modify after loading YAML (hypothetical)
# config = load_yaml_config("my_scenario.yaml")
# config.sim_config.sim_time_in_secs = 50.0  # Quick override
# env = make_robot_env(config=config)
```

---

## Configuration Modules

### Canonical Module (Recommended)

**Module**: `robot_sf/gym_env/unified_config.py`

**Status**: ✅ **Active** - Use for all new code

**Classes**:
- `BaseSimulationConfig` - Core simulation configuration shared by all environments
- `RobotSimulationConfig` - Robot-based environments (extends `BaseSimulationConfig`)
- `ImageRobotConfig` - Robot with image observations (extends `RobotSimulationConfig`)
- `PedestrianSimulationConfig` - Pedestrian-adversary environments (extends `RobotSimulationConfig`)

**Purpose**: Provides a unified, consistent configuration hierarchy that eliminates duplication and clearly separates concerns.

### Legacy Modules (Deprecated)

**Module**: `robot_sf/gym_env/env_config.py`

**Status**: ⚠️ **Deprecated** - Will be removed in a future version

**Classes**:
- `BaseEnvSettings` → Use `BaseSimulationConfig` instead
- `RobotEnvSettings` → Use `RobotSimulationConfig` instead
- `EnvSettings` → Use `RobotSimulationConfig` instead
- `PedEnvSettings` → Use `PedestrianSimulationConfig` instead

**Deprecation Timeline**:
- **Current (Phase 2)**: Classes emit `DeprecationWarning` on instantiation
- **Migration Period**: Users should migrate to unified config classes
- **Future (Phase 3)**: Legacy classes will be removed after sufficient migration time

**Note**: Legacy classes remain fully functional with warnings. No immediate action required, but migration is recommended.

---

## Unified Config Classes

### BaseSimulationConfig

**Purpose**: Core simulation configuration shared by all environment types.

**Key Fields**:
- `sim_config`: `SimulationSettings` - Simulation time, difficulty, pedestrian density
- `map_pool`: `MapDefinitionPool` - Available maps for environment
- `map_id`: `str | None` - Optional deterministic map selection from the pool
- `lidar_config`: `LidarScannerSettings` - LiDAR sensor configuration
- `render_scaling`: `int | None` - Optional UI/render scaling factor
- `backend`: `str` - Simulation backend selector (default: "fast-pysf")
- `sensors`: `list[dict]` - Sensor wiring configuration

**Usage**:
```python
from robot_sf.gym_env.unified_config import BaseSimulationConfig

config = BaseSimulationConfig()
# Rarely instantiated directly; typically extended by RobotSimulationConfig
```

### RobotSimulationConfig

**Purpose**: Configuration for robot-based navigation environments.

**Extends**: `BaseSimulationConfig`

**Additional Fields**:
- `robot_config`: `DifferentialDriveSettings | BicycleDriveSettings` - Robot kinematics
- `use_image_obs`: `bool` - Enable/disable image observations (default: False)
- `peds_have_static_obstacle_forces`: `bool` - Enable pedestrian-obstacle forces (default: True)
- `peds_have_robot_repulsion`: `bool | None` - Enable pedestrian-robot repulsion (defaults to sim_config)
- `peds_have_obstacle_forces`: `bool | None` - Deprecated alias for static obstacle forces

**Usage**:
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.gym_env.environment_factory import make_robot_env

config = RobotSimulationConfig()
config.peds_have_static_obstacle_forces = True  # Enable ped-obstacle physics
config.peds_have_robot_repulsion = True  # Enable ped-robot repulsion

env = make_robot_env(config=config)
```

**Methods**:
- `robot_factory()` - Creates robot instance based on `robot_config` type

### ImageRobotConfig

**Purpose**: Configuration for robot environments with image observations.

**Extends**: `RobotSimulationConfig`

**Additional Fields**:
- `image_config`: `ImageSensorSettings` - Image sensor configuration
- `use_image_obs`: `bool` - Automatically set to `True` (overrides parent default)

**Usage**:
```python
from robot_sf.gym_env.unified_config import ImageRobotConfig
from robot_sf.gym_env.environment_factory import make_image_robot_env

config = ImageRobotConfig()
# use_image_obs is True by default

env = make_image_robot_env(config=config)
```

### PedestrianSimulationConfig

**Purpose**: Configuration for pedestrian-adversary environments (ego pedestrian).

**Extends**: `RobotSimulationConfig`

**Additional Fields**:
- `ego_ped_config`: `UnicycleDriveSettings` - Ego pedestrian kinematics

**Usage**:
```python
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.gym_env.environment_factory import make_pedestrian_env

config = PedestrianSimulationConfig()
env = make_pedestrian_env(robot_model=trained_model, config=config)
```

**Methods**:
- `pedestrian_factory()` - Creates pedestrian instance based on `ego_ped_config`

---

## YAML Configuration

### Mapping YAML to Unified Classes

YAML configuration files map directly to unified config class fields. The structure mirrors the dataclass hierarchy.

**Example**: YAML for `RobotSimulationConfig`

```yaml
# Maps to RobotSimulationConfig fields
sim_config:  # SimulationSettings
  sim_time_in_secs: 200.0
  difficulty: 2
  peds_speed_mult: 1.3

map_pool:  # MapDefinitionPool
  map_name: "corridor"
map_id: "corridor"  # Optional deterministic selection from map_pool

lidar_config:  # LidarScannerSettings
  num_rays: 20
  max_range: 10.0

robot_config:  # DifferentialDriveSettings
  max_lin_vel: 1.5
  max_ang_vel: 1.0

peds_have_static_obstacle_forces: true
peds_have_robot_repulsion: true
```

### Loading YAML Configs

YAML configs are typically loaded by benchmark runners or training scripts:

```python
import yaml
from robot_sf.gym_env.unified_config import RobotSimulationConfig

# Load YAML (simplified - actual implementation may vary)
with open("configs/scenarios/my_scenario.yaml") as f:
    config_dict = yaml.safe_load(f)

# Create config from dict (simplified)
# Actual implementation may use validation/parsing helpers
config = RobotSimulationConfig(**config_dict)

env = make_robot_env(config=config)
```

**Note**: YAML loading and validation logic may reside in scenario generators or benchmark utilities. Refer to `robot_sf/benchmark/` for production usage examples.

---

## External Configuration

### fast-pysf Backend Configuration

The `fast-pysf` pedestrian simulation backend has its own configuration system, managed externally.

**Relationship**:
- robot_sf configs contain parameters for the **robot, environment, and sensors**
- fast-pysf configs control **pedestrian dynamics and social forces**
- The two systems are **loosely coupled** via the backend selector (`backend="fast-pysf"`)

**Pass-through behavior**:
- Some pedestrian-related parameters in `SimulationSettings` (e.g., `peds_speed_mult`, `ped_radius`) are passed to the fast-pysf backend
- Direct fast-pysf configuration is handled by the `FastPysfWrapper` in `robot_sf/sim/`

**Example**:
```python
config = RobotSimulationConfig()
config.backend = "fast-pysf"  # Select fast-pysf backend
config.sim_config.peds_speed_mult = 1.5  # Passed to fast-pysf

env = make_robot_env(config=config)
```

**For advanced fast-pysf configuration**, see `docs/fast_pysf_wrapper.md`.

### Other External Configs

- **StableBaselines3 training configs**: Managed separately in training scripts
- **Benchmark runner configs**: Scenario definitions and episode parameters
- **Visualization configs**: Render options and recording settings

---

## Best Practices

### When to Use Each Configuration Level

**Code Defaults**:
- ✅ Sensible starting values that work for most users
- ✅ Parameters that rarely change across experiments
- ✅ Values that define "standard" behavior

**YAML Files**:
- ✅ Reusable experiment configurations
- ✅ Benchmark scenarios shared across runs
- ✅ Project-specific presets
- ✅ Version-controlled parameter sets for reproducibility

**Runtime Parameters**:
- ✅ One-off experiments or debugging
- ✅ Parameter sweeps (vary one parameter across runs)
- ✅ Quick overrides during development
- ✅ Interactive exploration in notebooks

### Configuration Best Practices

1. **Prefer explicit over implicit**: Always specify config objects rather than relying on default `None` parameters in factory functions
2. **Version your YAML configs**: Commit scenario files to track experiment history
3. **Document custom configs**: Add comments in YAML files explaining non-obvious parameter choices
4. **Use type-safe classes**: Leverage dataclass type hints to catch errors early
5. **Test with minimal configs**: Start with defaults, add overrides incrementally
6. **Avoid deep nesting**: Keep configuration structure flat where possible for readability

### Common Pitfalls

❌ **Don't**: Modify config objects after env creation (changes won't apply)
```python
config = RobotSimulationConfig()
env = make_robot_env(config=config)
config.peds_have_static_obstacle_forces = True  # Too late! Env already created
```

✅ **Do**: Create config, set all parameters, then create env
```python
config = RobotSimulationConfig()
config.peds_have_static_obstacle_forces = True  # Before env creation
env = make_robot_env(config=config)  # Config is frozen into env
```

❌ **Don't**: Mix legacy and unified configs in the same codebase without plan
```python
# Confusing and hard to maintain
from robot_sf.gym_env.env_config import EnvSettings  # Legacy
from robot_sf.gym_env.unified_config import RobotSimulationConfig  # Unified
```

✅ **Do**: Use unified configs consistently, migrate legacy usage incrementally
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig
config = RobotSimulationConfig()
```

---

## Migration Guide

### Overview

Legacy config classes (`EnvSettings`, `PedEnvSettings`, `RobotEnvSettings`, `BaseEnvSettings`) are deprecated in favor of the unified configuration hierarchy. This guide shows how to migrate existing code.

**Migration is optional but recommended**:
- Legacy classes continue to work with `DeprecationWarning`
- No breaking changes in current version
- Future versions will remove legacy classes (Phase 3)

### EnvSettings → RobotSimulationConfig

**Legacy (deprecated)**:
```python
from robot_sf.gym_env.env_config import EnvSettings

config = EnvSettings()
# ... use config ...
```

**Unified (recommended)**:
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig

config = RobotSimulationConfig()
# ... use config ... (same API, same fields)
```

**Field Mapping**: Direct 1:1 equivalence
- `sim_config` → `sim_config`
- `map_pool` → `map_pool`
- `lidar_config` → `lidar_config`
- `robot_config` → `robot_config`
- `render_scaling` → `render_scaling`

**Behavioral Changes**: None - identical runtime behavior

### PedEnvSettings → PedestrianSimulationConfig

**Legacy (deprecated)**:
```python
from robot_sf.gym_env.env_config import PedEnvSettings

config = PedEnvSettings(
    ego_ped_config=my_ped_config
)
```

**Unified (recommended)**:
```python
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig

config = PedestrianSimulationConfig(
    ego_ped_config=my_ped_config
)
```

**Field Mapping**: Direct 1:1 equivalence
- All `RobotEnvSettings` fields (inherited) → `RobotSimulationConfig` fields
- `ego_ped_config` → `ego_ped_config`

**Behavioral Changes**: None - identical runtime behavior

### RobotEnvSettings → RobotSimulationConfig

**Legacy (deprecated)**:
```python
from robot_sf.gym_env.env_config import RobotEnvSettings

config = RobotEnvSettings(
    use_image_obs=True
)
```

**Unified (recommended)**:
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig

config = RobotSimulationConfig(
    use_image_obs=True
)
```

**Field Mapping**: Direct 1:1 equivalence
- All fields identical between legacy and unified

**Behavioral Changes**: None - identical runtime behavior

### Common Migration Scenarios

#### Scenario 1: Basic Robot Environment

**Before**:
```python
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.environment_factory import make_robot_env

config = EnvSettings()
env = make_robot_env(config=config)
```

**After**:
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.gym_env.environment_factory import make_robot_env

config = RobotSimulationConfig()
env = make_robot_env(config=config)
```

#### Scenario 2: Custom Simulation Settings

**Before**:
```python
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.sim.sim_config import SimulationSettings

config = EnvSettings(
    sim_config=SimulationSettings(
        sim_time_in_secs=100.0,
        difficulty=3
    )
)
```

**After**:
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.sim.sim_config import SimulationSettings

config = RobotSimulationConfig(
    sim_config=SimulationSettings(
        sim_time_in_secs=100.0,
        difficulty=3
    )
)
```

#### Scenario 3: Pedestrian Environment

**Before**:
```python
from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.ped_ego.unicycle_drive import UnicycleDriveSettings

config = PedEnvSettings(
    ego_ped_config=UnicycleDriveSettings(max_lin_vel=2.0)
)
```

**After**:
```python
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.ped_ego.unicycle_drive import UnicycleDriveSettings

config = PedestrianSimulationConfig(
    ego_ped_config=UnicycleDriveSettings(max_lin_vel=2.0)
)
```

### Migration Checklist

- [ ] Identify all uses of legacy config classes in your codebase
- [ ] Update import statements to use `unified_config` module
- [ ] Change class names to unified equivalents (field names unchanged)
- [ ] Run tests to verify identical behavior
- [ ] Commit changes with note about deprecation migration

### Behavioral Differences

**None identified** - Legacy and unified configs are designed to be functionally equivalent. Field names, types, and validation logic are identical. The only difference is the import path and class name.

If you encounter any behavioral differences during migration, please report as a bug.

---

## Future Work

### Phase 3: Legacy Config Consolidation (Deferred)

**Planned**: Remove legacy config classes from codebase entirely

**Timeline**: To be determined (depends on user migration progress)

**Prerequisites**:
- Sufficient migration period (minimum 2-3 release cycles)
- User communication about removal timeline
- Zero usage of legacy classes in examples and documentation

**Scope**:
- Delete `EnvSettings`, `PedEnvSettings`, `RobotEnvSettings`, `BaseEnvSettings` from `env_config.py`
- Remove deprecation warnings (no longer needed)
- Update all internal code to use unified configs exclusively
- Bump major version (breaking change)

**Rationale**: Consolidation reduces maintenance burden, eliminates duplication, and simplifies the configuration API. However, it requires careful coordination to avoid breaking user code. Deferring to Phase 3 allows gradual migration without disruption.

---

## References

- **Development Guide**: [docs/dev_guide.md](../dev_guide.md) - Configuration hierarchy section
- **Environment Overview**: [docs/ENVIRONMENT.md](../ENVIRONMENT.md) - Environment usage patterns
- **Factory Functions**: `robot_sf/gym_env/environment_factory.py` - Environment creation
- **Unified Configs**: `robot_sf/gym_env/unified_config.py` - Canonical config classes
- **Legacy Configs**: `robot_sf/gym_env/env_config.py` - Deprecated classes (with warnings)
- **Simulation Settings**: `robot_sf/sim/sim_config.py` - Core simulation parameters
- **fast-pysf Wrapper**: [docs/fast_pysf_wrapper.md](./fast_pysf_wrapper.md) - Pedestrian backend integration
