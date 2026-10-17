# Opt-in Segway T60 capsule collision diagnostic

The [30-degree](../../configs/robots/t60_capsule_bicycle_30deg_v1.yaml) and
[45-degree](../../configs/robots/t60_capsule_bicycle_45deg_v1.yaml) fragments select
instantaneous capsule collision checking. Copy their `robot_config` mapping into
an explicitly authored diagnostic scenario before calling
`build_robot_config_from_scenario`. They never select themselves in a released
matrix. The existing `t60_bicycle_*` fragments remain covering-disc diagnostics.
The [legacy reference robot](../../configs/robots/legacy_reference_robot_v1.yaml)
explicitly selects the unchanged 1.0 m disc and default differential drivetrain.

The body dimensions and driving cases follow author decisions in
[#10093](https://github.com/ll7/robot_sf_ll7/issues/10093) and
[#10098](https://github.com/ll7/robot_sf_ll7/issues/10098): total length 1.183 m,
full width 0.478 m, centre segment 0.705 m, circular end radius 0.239 m,
wheelbase estimate 0.90 m, steering limits 30/45 degrees, and riderless speed
proxy 1.34 m/s. These decisions are inputs, not newly verified manufacturer
steering/wheelbase/speed specifications. Acceleration and braking retain the
existing proxy settings. Reverse and creep are off in these fragments.

`capsule: {length: 1.183, width: 0.478}` is supported on bicycle settings only.
The pose denotes the body centre; the segment follows its live heading. This
is a modelling convention, not a measured axle/centre offset. The exact
capsule's smallest centred covering disc has radius 0.5915 m. The profile keeps
`radius: 0.64` as a conservative proxy for unmigrated consumers, independently
of the true cap radius. An undersized proxy or malformed capsule raises an
error instead of silently falling back to a disc. The selector is included
in the environment config hash only when enabled; default dataclass payloads
and identities retain their historical layout.

Continuous occupancy uses analytic point–segment and segment–segment surface
gaps for walls, pedestrians and robot bodies, including crossing, parallel and
degenerate segments. Robot–robot checks use both live headings, and ego
pedestrian checks see the same capsule. The whole capsule must lie inside map
bounds. Surface contact counts as collision, with a 1e-12 m roundoff allowance
on the capsule path. Disc checks retain their historical implementation,
including their centre-only bounds convention. Checks occur at sampled poses;
this is neither swept collision detection nor physical non-interpenetration.

## Remaining consumers and validation boundary

This first building block does not establish a capsule-aware planner or a
complete reference vehicle. Planner clearance and candidate motion, pedestrian
repulsion, spawn/goal feasibility, lidar mounts and other-robot ray geometry,
occupancy rasterization, observation footprint metadata and rendering still use
existing contracts. Capsule trajectory metrics need live heading, explicit body
metadata and versioned definitions; the parked metrics work in
[#10231](https://github.com/ll7/robot_sf_ll7/pull/10231) concerns disc surface
metrics. Do not use disc metric outputs to claim capsule safety or rankings.

The bicycle coupled execution projection is already on main. Plant-aware
prediction candidates are separate work in
[#10336](https://github.com/ll7/robot_sf_ll7/pull/10336), and the matched diagnostic
remains [#10109](https://github.com/ll7/robot_sf_ll7/issues/10109). Projection of a
command alone does not prove that the planner scored the executed trajectory.
Full capsule planner support requires explicit admission checks and geometry/
kinematics migration. Learned disc checkpoints require retraining and validation.
Contact callbacks are separate from sensor callbacks: other capsule robots remain
visible to lidar through the explicit covering-disc approximation.

Run the focused physical witnesses with:

```sh
uv run pytest tests/test_t60_capsule_collision.py tests/test_capsule_footprint.py \
  tests/test_bicycle_drive.py tests/test_bicycle_planner_physics.py \
  tests/test_bicycle_creep_safety.py tests/test_classic_planner_adapter.py -q -n 0
```

The surface-distance cross-check uses dev seeds 1001–1200 only. Tests protect
physical contact values and production config/factory wiring, with no production
test seam. Existing occupancy tests exercise discs but omit elongated bodies,
heading-dependent contact and capsule identity. New regression witnesses can
run against base main using its existing factories and fail on physical
assertions or the missing validation, rather than missing imports. No frozen
0.0.8 configuration, artifact, metric definition or scenario is changed.

## Planner admission

Every current capsule planner combination is refused. The map-runner checks
resolved robot settings before checkpoint/observation loading, its public policy
builder checks both supplied and embedded robot configs before dispatch, and
adapter `bind_env` methods check both authored settings and the live robot plant.
An unsupported combination raises `UnsupportedRobotFootprintError` naming the
planner and the missing geometry/rollout contract; unknown planners fail closed.
Grid/visibility global route attachment is also refused, including maps already
carrying a planner. There is no automatic covering-disc substitution.

| Planner family | Missing capsule/bicycle capability |
| --- | --- |
| Goal seeker, social force | Heading-dependent body clearance and verified bicycle planning |
| ORCA/RVO2/HRVO; placeholder RVO | Circular velocity obstacles / holonomic commands do not establish capsule bicycle support |
| Sampling, DWA/risk DWA, prediction, MPPI variants, safety barrier | Circular clearance and unverified capsule candidate/plant trajectories |
| Hybrid rule / hybrid RL / guarded PPO; selector and group wrappers | Children and safety scorers lack the required body/motion contract |
| SACADRL, SocNav/CrowdNav/GST bridges, SICNav, DR-MPC | No verified capsule/bicycle adapter contract |
| Existing PPO/recurrent checkpoints | Disc observations/training and unvalidated body/dynamics transfer |
| Classic grid and visibility global routes | No SE(2) capsule/bicycle feasibility guarantee |

Native-command *benchmark* construction is refused too: it does not establish a
capsule planning/metric contract. Manual plant diagnostics remain possible through
`make_robot_env` and `PlannerActionAdapter`, which converts commands without
claiming planner support. Explicit disc proxy diagnostics remain available under
their existing command-projection contracts. Those contracts must not be described
as full native bicycle planner support. Guards can be relaxed per planner only
after footprint-aware clearance, executed candidate motion, observation/metric
provenance and relevant failure witnesses are verified.

```sh
uv run pytest tests/planner/test_capsule_planner_contract.py \
  tests/benchmark/test_planner_command_contract.py -q -n 0
```

The guard witnesses protect rejection before optional dependency/fallback loading,
all direct binding methods, resolved episode settings and global route mutation.
Existing compatibility tests only cover command/observation contracts. They do not
admit an elongated body. Test spies target existing dispatch/loading methods;
no test-only production seam is added. The explicit disc control still constructs
and binds a real planner.
