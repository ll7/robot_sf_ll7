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
config.backend = "fast-pysf"  # or a future backend
config.peds_have_obstacle_forces = True
config.sensors = [
    {"name": "lidar", "params": {"range": 10.0, "fov": 180}},
]

env = make_robot_env(config=config)
state, info = env.reset(seed=123)
```

## Notes
- Backends are resolved by name via a registry. If unknown, you get a clear KeyError with suggestions.
- Sensors should be registered in a central sensor registry; factories must be pure wrt config.
- For maps, prefer paths under `maps/` and validate existence early.
