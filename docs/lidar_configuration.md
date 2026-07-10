# LiDAR Configuration Reference

Robot environments use a 272-ray LiDAR scan by default, so this page gives the
numbers to use when configuring, comparing, or describing an environment.

## Canonical robot default

`RobotSimulationConfig` and `BaseSimulationConfig` create a
`LidarScannerSettings` instance unless a caller supplies `lidar_config`. The
current default is:

| Setting | Value | Meaning |
| --- | --- | --- |
| `num_rays` | 272 | Equally spaced range readings in each scan. |
| `visual_angle_portion` | `1.0` | Full 360-degree field of view. |
| `max_scan_dist` | 10 m | Maximum range returned by a ray, in map/world units. |
| Angular spacing | about 1.32 degrees per ray | `360 / 272`; this is not a one-degree sensor. |
| `scan_noise` | `[0.005, 0.002]` | Per-ray loss and corruption probabilities, respectively. |
| `detect_other_robots` | `True` | Dynamic objects exposed by the occupancy source are treated as circular obstacles. |

The value 272 is retained because it is the established default observation
dimension used by the repository's LiDAR-facing tooling and learned-policy
input contracts; it is not a hardware-calibration claim or a one-degree-ray
approximation.

The implementation authority is
[`LidarScannerSettings`](../robot_sf/sensor/range_sensor.py) and the default
environment wiring is in
[`BaseSimulationConfig`](../robot_sf/gym_env/unified_config.py).

## Ego-pedestrian variant

`LidarScannerSettings.ego_pedestrian_lidar()` is an explicit alternate
configuration for the ego pedestrian; it is not the robot-environment default.

| Setting | Value |
| --- | --- |
| `num_rays` | 272 |
| Field of view | 120 degrees (`visual_angle_portion = 1 / 3`) |
| `max_scan_dist` | 30 m |
| Angular spacing | about 0.44 degrees per ray (`120 / 272`) |
| Noise and dynamic-object detection | Inherit the canonical `LidarScannerSettings` defaults above. |

## Benchmark interpretation

Benchmark and policy observation overrides do not expose a LiDAR-configuration
override key. Therefore, a standard benchmark run uses the canonical robot
default unless its environment configuration is constructed differently by the
calling code. Record an explicit custom `lidar_config` with the run whenever a
comparison changes these values; the built-in run provenance records LiDAR
noise but not the full sensor configuration.

This reference describes repository defaults only. It does not establish a
hardware sensor model, a calibrated noise model, or benchmark-performance
evidence.
