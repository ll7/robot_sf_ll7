# Holonomic Action Contract

This document defines the benchmark-facing holonomic robot command contract in Robot SF.
It is the source of truth for frame semantics, heading behavior, and planner adapter boundaries.

## Runtime Modes

`HolonomicDriveRobot` supports two command modes:

- `vx_vy`
- `unicycle_vw`

Both modes keep a full robot pose `((x, y), heading)`. Holonomic motion does not mean
"orientation-free." It means the translational motion model is not constrained to move only along
the current heading.

## `vx_vy` Mode

Action shape:

- `(2,)`

Action meaning:

- `action[0] = vx`
- `action[1] = vy`

Frame:

- `vx` and `vy` are **world-frame translational velocities**
- they are **not** body-frame forward/lateral velocities

Execution rule:

- the robot translates by `x += vx * dt`, `y += vy * dt`
- speed is clipped to `max_speed`

Heading rule:

- if `sqrt(vx^2 + vy^2) > 0`, heading becomes `atan2(vy, vx)`
- if `vx == 0` and `vy == 0`, heading remains unchanged

Implication:

- "forward direction" in `vx_vy` mode is the current heading state, and while the robot is moving
  it aligns with the world-frame velocity direction
- there is no independent yaw command in this mode

## `unicycle_vw` Mode

Action shape:

- `(2,)`

Action meaning:

- `action[0] = v`
- `action[1] = omega`

Frame:

- `v` is scalar forward speed along the robot heading
- `omega` is yaw rate

Execution rule:

- heading integrates first by `heading += omega * dt`
- translational velocity is then resolved into world coordinates from the updated heading

This mode keeps holonomic robot state and observations, but the command contract is unicycle-like.

## Observation Contract

Holonomic robots still use the standard SocNav structured observation contract in
[`docs/dev/observation_contract.md`](./observation_contract.md).

The key fields for holonomic interpretation are:

- `robot.heading`: always present
- `robot.speed`: `(linear_speed, angular_speed)`
- `robot.velocity_xy`: explicit world-frame translational velocity
- `robot.angular_velocity`: explicit yaw rate

Important distinction:

- `robot.speed` is **not** `vx, vy`
- `robot.velocity_xy` is the explicit world-frame translational velocity

## Benchmark Bridge Contract

The benchmark runner accepts two policy command shapes:

- legacy unicycle tuple `(v, omega)`
- structured holonomic command

Structured holonomic command payload:

```python
{
    "command_kind": "holonomic_vxy_world",
    "vx": <float>,
    "vy": <float>,
}
```

Semantics:

- `vx` and `vy` are world-frame translational velocities
- for a holonomic robot in `vx_vy` mode, this is forwarded directly to the environment action
- for differential-drive or bicycle robots, Robot SF converts this world velocity into the
  existing `(v, omega)` compatibility path

## ORCA Contract

Native ORCA semantics are closest to a world-frame 2D preferred/selected velocity.

Robot SF therefore uses two distinct runtime paths:

- holonomic `vx_vy` benchmark:
  - in-repo ORCA and `social_navigation_pyenvs_orca` both expose a world-frame velocity vector
  - the benchmark forwards that velocity directly through the structured holonomic command
- differential-drive benchmark:
  - ORCA world-frame velocity is converted to `(v, omega)` through the existing
    heading-safe compatibility adapter

This avoids the redundant round-trip:

- world velocity -> `(v, omega)` -> world velocity

for the holonomic benchmark path.

## What Is Not Allowed To Be Ambiguous

- `vx_vy` means world-frame translational velocity, not body-frame velocity
- holonomic robots still have heading in state and observation
- `robot.speed` is `(linear_speed, angular_speed)`, not `(vx, vy)`
- in `vx_vy` mode, heading follows velocity direction while moving
- any planner benchmarked as holonomic must declare whether it is:
  - direct world-velocity,
  - unicycle-projected,
  - or unavailable under the holonomic contract
