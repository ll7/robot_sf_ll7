# Quickstart: Select backends and wire sensors

This quickstart shows the programmatic way (preferred) to select a simulator backend and register sensors using the unified config.

## TL;DR
- Use factory functions, not direct env classes
- Configure via `robot_sf.gym_env.unified_config`
- Keep it scriptable; avoid CLI-only knobs

## Example
```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig

config = RobotSimulationConfig()
config.backend = "fast-pysf"  # or "dummy" for testing
config.peds_have_obstacle_forces = True

env = make_robot_env(config=config)
state, info = env.reset(seed=123)
```

## Adding a custom sensor

Register sensors via config to augment observations without modifying SensorFusion:

```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig

config = RobotSimulationConfig()
config.backend = "fast-pysf"
config.sensors = [
    {
        "type": "dummy_constant",  # registered sensor name
        "name": "bias",            # observation key will be "custom.bias"
        "value": [0.5, 0.25],
        "shape": [2],
        "dtype": "float32",
        "space": {"shape": [2], "low": 0.0, "high": 1.0},
    }
]

env = make_robot_env(config=config)
obs, info = env.reset()
# obs now includes 'custom.bias' alongside 'drive_state' and 'rays'
print(obs.keys())  # dict_keys(['drive_state', 'rays', 'custom.bias'])
```

## Error Handling Policy

The system follows a consistent error policy across all components:

### Fatal Errors (Required Resources)
Missing or invalid required resources raise `RuntimeError` with **actionable remediation**:

```python
# Example: Missing map file
RuntimeError: Map file not found: maps/svg_maps/my_map.svg

Remediation: Place SVG map at 'maps/svg_maps/my_map.svg' or check available maps in maps/svg_maps/
```

```python
# Example: Missing PPO model (without fallback)
RuntimeError: PPO model file not found: model/ppo_model.zip

Remediation: Place model at 'model/ppo_model.zip' or check available models in model/ directory.
Download from releases or train with scripts/training_ppo.py
```

### Soft-Degrade Warnings (Optional Components)
Optional component failures log `WARNING` and continue with fallback behavior:

```python
# Example: PPO model missing with fallback enabled
WARNING: Optional component 'PPO model' issue: Model not found at model/ppo_model.zip.
Fallback: will use fallback-to-goal navigation
```

### Unknown Names (Configuration Errors)
Invalid backend or sensor names raise `KeyError` listing available alternatives:

```python
# Example: Unknown sensor
KeyError: Unknown sensor 'invalid_sensor'. Available sensors: dummy_constant, lidar
```

### Best Practices
- **Maps**: Validate map paths early; missing maps raise fatal errors
- **Models**: Enable `fallback_to_goal=True` in PPOPlannerConfig for robustness
- **Sensors**: Check `robot_sf.sensor.registry.list_sensors()` for available types
- **Backends**: Check `robot_sf.sim.registry.list_backends()` for available simulators

## Configuration Validation

All environment factories automatically validate configuration before instantiation:

### Automatic Validation
```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig

config = RobotSimulationConfig()
config.backend = "invalid_backend"  # Typo!

# Raises KeyError: Unknown backend 'invalid_backend'. Available backends: dummy, fast-pysf
env = make_robot_env(config=config)
```

### Validation Rules
1. **Unknown Keys**: Strict mode rejects unknown configuration keys (helps catch typos)
2. **Backend Names**: Must match registered backend (checks `robot_sf.sim.registry`)
3. **Sensor Types**: Must match registered sensor types (checks `robot_sf.sensor.registry`)
4. **Conflicts**: Detects mutually exclusive options (e.g., `use_image_obs` without `image_config`)

### Resolved Config Logging
Every environment creation logs the resolved configuration for reproducibility:

```python
config = RobotSimulationConfig()
config.backend = "fast-pysf"
config.sensors = [{"type": "dummy_constant", "name": "bias", ...}]

env = make_robot_env(config=config)
# INFO: Resolved config: type=RobotSimulationConfig backend=fast-pysf sensors=1
# DEBUG: Full resolved config: {'backend': 'fast-pysf', 'sensors': [...], ...}
```

### Manual Validation (Advanced)
For pre-flight checks or testing:

```python
from robot_sf.gym_env.config_validation import validate_config

config = RobotSimulationConfig()
config.backend = "fast-pysf"

# Validate before expensive setup
validate_config(config, strict=True)  # Raises on any issue

# Get resolved config dict for logging/debugging
from robot_sf.gym_env.config_validation import get_resolved_config_dict
resolved = get_resolved_config_dict(config)
print(resolved)  # Full config as dict with all defaults resolved
```

## Notes
- **Backends**: Resolved by name via registry (`robot_sf.sim.registry`). Unknown names raise `KeyError` with suggestions.
- **Sensors**: Registered in `robot_sf.sensor.registry`. Each sensor config must include:
  - `type`: Registered sensor name
  - `name`: Key for observation (output as `custom.<name>`)
  - `space`: Gym Box spec (`shape`, `low`, `high`) to extend observation space
- **Maps**: Prefer paths under `maps/` and validate existence early.
- **Validation**: Automatic at env creation; use `validate_config()` for early checks.
