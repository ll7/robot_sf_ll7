# Waypoint Noise For Route Generalization

Waypoint sampling can optionally add Gaussian noise to intermediate route waypoints to reduce
route memorization during training.

## Config

Use `RobotSimulationConfig.navigation_settings`:

- `waypoint_noise_enabled`: enable/disable noise (default: `false`)
- `waypoint_noise_std`: Gaussian standard deviation in meters (default: `0.0`)

Noise is applied only to route waypoints. Sampled spawn and sampled goal positions are unchanged.

## Recommended Ranges

- `0.02` to `0.05` m: low-noise regularization (safe starting point)
- `0.05` to `0.15` m: stronger domain randomization for robustness studies
- `> 0.15` m: use with caution; may distort narrow-route maps

## Example

```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.navigation import NavigationSettings

cfg = RobotSimulationConfig(
    navigation_settings=NavigationSettings(
        waypoint_noise_enabled=True,
        waypoint_noise_std=0.05,
    )
)
```
