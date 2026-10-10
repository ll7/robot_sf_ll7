# Assumed sidewalk delivery robot

[Profile](../../configs/robots/assumed_sidewalk_delivery_v1.yaml) is an opt-in
**diagnostic assumption set** for 0.1.0, requested in
[issue #10189](https://github.com/ll7/robot_sf_ll7/issues/10189).
It does not change any default or select itself in a released matrix. Selecting
this profile does not reproduce a particular commercial robot.

| Override | Value | Status and basis |
| --- | --- | --- |
| `type` | differential_drive | Assumption: simulator drive model |
| `radius` | 0.31 m | Assumption: half of 0.62 m exterior width; a width proxy, not a covering disc |
| `max_linear_speed` | 1.2 m/s | Assumption: operational cap |
| `max_linear_accel` | 0.5 m/s² | Assumption: forward acceleration |
| `max_linear_decel` | 0.8 m/s² | Assumption: braking magnitude |
| `max_angular_speed` | 1.0 rad/s | Assumption: yaw cap |
| `allow_backwards` | false | Assumption: no reverse |
| `limited_reverse` | false | Assumption: no limited reverse |
| `control_latency_s` | 0.2 s | Assumption: command delay, using the existing simulator action queue |

The **only manufacturer-attributed dimensional input** is Segway E1 exterior
width 620 mm in the manufacturer-branded
[Segway E1 Robot Specifications](https://cdn.robotshop.com/rbm/e40a85cf-9ab9-4664-b045-f28aab26e201/e/e8feeb22-caa5-449c-8e9b-8e9db9fbb9d5/2963d5f0_e1-delivery-parameters.pdf),
hosted by RobotShop (accessed 2026-10-08). The
[manufacturer product page](https://robotics.segway.com/e1/) identifies the E1.
Neither source establishes the assumed acceleration, braking, yaw, latency or
operating speed above. A 0.31 m disc does not cover an elongated body: capsule
collision physics remains out of scope. Unspecified wheel dimensions and angular
acceleration retain simulator defaults, also unvalidated for a delivery robot.

## Opt-in use and radius sensitivity

The file is a robot override fragment, not a scenario manifest. Copy its
`robot_config` mapping into the diagnostic scenario before the normal loader:

```python
from copy import deepcopy
from pathlib import Path
import yaml
from robot_sf.training.scenario_loader import build_robot_config_from_scenario

profile = yaml.safe_load(Path("configs/robots/assumed_sidewalk_delivery_v1.yaml").read_text())
diagnostic = deepcopy(scenario)
diagnostic["robot_config"] = deepcopy(profile["robot_config"])
config = build_robot_config_from_scenario(diagnostic, scenario_path=scenario_path)
```

Use this plant in the footprint/radius sensitivity sweep
([#6600](https://github.com/ll7/robot_sf_ll7/issues/6600)): vary only `radius`
across 0.31, 0.5 and 1.0 m while holding the remaining profile values, scenario,
planner and dev seeds 1001-1030 fixed. Keep the unchanged default plant as a
separate comparator; comparing it directly to the whole profile confounds radius
with speed, acceleration, braking and latency. Record the selected override
mapping in every diagnostic run. No sweep results or planner rankings are claimed
by this profile addition.

An optional braking sensitivity case copies the same mapping and sets
`max_linear_decel: 0.4` m/s² (**assumption**). At the nominal 0.1 s timestep,
0.2 s latency is two action-queue steps. Other timesteps round latency up to the
next complete step; `sim_config.action_latency_metadata()` reports the effective
delay. A simultaneous nonzero `simulation_config.action_latency_steps` is rejected
by the existing latency validation. `control_latency_s` takes precedence over a
millisecond latency in `simulation_config` when explicitly selected.

An opt-in profile alone does not change released numbers. Changing defaults or a
released matrix remains subject to the behaviour-change gate.
