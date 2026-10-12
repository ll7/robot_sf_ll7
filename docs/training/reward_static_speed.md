# Opt-in speed cost near static geometry

Built-in route-completion and social-quality rewards accept a `static_speed` weight,
defaulting to **0.0**. Existing configs keep the same scalar reward. Enable only in a new
training configuration, for example:

```yaml
env_factory_kwargs:
  reward_name: route_completion_v3
  reward_kwargs:
    weights:
      static_speed: -0.5
```

The unweighted cost is `abs(physical_speed) * clip(1 - static_clearance / 1 m, 0, 1)`.
The distance is surface clearance from the robot disc to the nearest static wall, obstacle,
or map boundary, using collision geometry; obstacle interiors and negative clearance count
as zero clearance. Static pedestrians are excluded. Cost is zero at rest and at clearance
of at least 1 m, and grows linearly with speed and proximity below 1 m. Reverse speed also
incurs cost. This is a shaping signal, not a braking controller or collision guarantee.

`RobotEnv` computes `static_obstacle_clearance` (m) and `robot_speed` (m/s) only when a built-in
reward opts in. Curricula enable metadata if any stage requests this weight; a zero-weight
stage still has zero cost. Custom rewards can supply these metadata fields themselves.

This changes future training behavior only when explicitly enabled, not benchmark evaluation
of existing checkpoints. Observations, actions, termination, canonical benchmark metrics and
frozen artifacts are unchanged. No training or evaluation campaign was run.
