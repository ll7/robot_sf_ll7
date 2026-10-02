# Limited reverse driving

<!-- AI-GENERATED NEEDS-REVIEW -->

Opt in through `robot_config` in a scenario, or the drive settings passed to
`RobotSimulationConfig`:

```yaml
robot_config:
  type: differential_drive
  limited_reverse: true
  max_reverse_speed: 0.5
```

`limited_reverse` selects the `limited_reverse.v1` plant. The independent reverse
cap defaults to 0.5 m/s; 0.3 m/s is also supported. The cap is a design choice,
not a standards-derived human-safety limit. Forward speed and acceleration settings
keep their existing meanings. Negative commands use the same signed acceleration
limits as positive commands; explicitly asymmetric braking remains supported.

Omitting the flag retains legacy behavior, including the historical full-speed
reverse of `allow_backwards: true`. The new InitVars do not add default fields to
legacy dataclass serialization. Environment identity and actuation provenance add
a versioned reverse block only for the opt-in plant. A cap without the flag does
not activate reverse. Both differential and bicycle drive settings support it;
unicycle `(v, omega)` command projection and the velocity-to-acceleration adapter
use the same cap. This does not change the pedestrian unicycle model.

The ORCA world-velocity adapter uses reverse as an opt-in escape, not whenever a
goal is behind it. It enters after three consecutive stationary steps (speed
below 0.05 m/s) with mean combined occupancy penalty at least 0.95 over two forward probes
just beyond the robot radius (0.15 and 0.30 m, shortened with lookahead), and a
heading error of at least 110 degrees, inclusive within numerical tolerance.
It exits when the forward heading error is no greater than the reverse heading
error (90 degrees), or after three consecutive clear forward probes.
The robot's own rear-aligned rotation increases forward error; the angular exit
responds to a swing in the goal or ORCA world target. This hysteresis prevents switching at every
90-degree crossing; free space retains the forward turn toward the goal.
Bound static geometry intersecting a 0.30 m forward footprint sweep, or an
observed pedestrian ahead of the robot centre overlapping that projected footprint, also establishes
obstruction without a grid; this supports canonical tracked-agent observations.
Backward travel turns toward the reverse travel axis and checks the rear
footprint and constant-velocity pedestrian prediction through a braking horizon.
The pedestrian rear check includes lateral and rear positions at each predicted
sample, including crossings from ahead; pedestrians still ahead of that sample
do not refuse movement away from a nose obstruction.
Absent rear geometry/grid data or an obstructed rear sweep restores the complete
forward command, including its turn. HRVO inherits this adapter.
This projection is an adapter heuristic; it does not preserve ORCA's holonomic
feasibility guarantee. Hybrid v4 adds negative dynamic-window/static-escape
candidates, predicts signed drive-limited rollouts, and checks braking toward
zero from either sign. Its existing continuous static and pedestrian gates apply
to backward trajectories too. Forward safety caps also bound reverse candidates.

Changed adapters: ORCA, inherited HRVO, hybrid v4, shared map-runner feasibility
binding, and `PlannerActionAdapter`. Legacy hybrid variants remain forward-only.
The unchanged goal/grid, social-force, risk-DWA, predictive-MPPI,
SocNav sampling/prediction, SACADRL, guarded-PPO guard, and learned/external planner
adapters do not add reverse candidates or read the new selector. A learned/native
negative command can only reverse if every upstream projection permits it;
retraining or lattice changes for those planners are separate work. Canonical
map-runner binding warns once, naming any adapter that is not reverse-aware,
when its live plant enables limited reverse. Hybrid v3 also gets this warning;
hybrid v4 and ORCA/HRVO do not.
Plain policies without a bound planner adapter do not emit an adapter warning.

The default LiDAR angle portion is 1.0 (a full 360-degree scan). Reverse guards
use observed pedestrian state and static map/grid geometry rather than assuming
that a rear sector is empty. The opt-in plant changes benchmark semantics:
comparisons against existing release results are diagnostic, not interchangeable.
The development gate compares off, 0.3, and 0.5 m/s on seeds 1001–1030 only,
with actor-free sweeps and paired failure classification. No frozen release
configuration or golden oracle is updated to make this feature pass.
