# Robot maneuver candidates

The maneuver portfolio produces alternative robot trajectories for a later planner to compare:
follow the route, pass on either side, creep, or brake to a stop. It does not select a trajectory
or enable a new planner in the benchmark runner.

## Generate an empty-world portfolio

```python
from robot_sf.nav.global_route import RouteGeometry
from robot_sf.planner.maneuver_candidates import generate_maneuver_candidates
from robot_sf.robot.dynamics import RobotDynamicsState

route = RouteGeometry(((0.0, 0.0), (10.0, 0.0)))
result = generate_maneuver_candidates(
    route,
    RobotDynamicsState(linear_speed=0.8),
    static_geometry=(),  # Explicitly known empty geometry for this example.
)
for candidate in result.candidates:
    print(candidate.maneuver.value, candidate.first_control)
report = result.report.as_dict()
```

Real callers must supply their planner-visible static geometry and runtime limits. Pass each
candidate's `action` directly to the canonical collision-risk estimator with the same timestep
and horizon; do not reconstruct trajectories from the first command alone.

## Contract and boundaries

Every candidate starts from the same robot state and uses the same timestep and horizon. Its
controls can be replayed through the canonical robot dynamics. The underlying
`CandidateAction` contains the initial position and one position after every control, so a
horizon of `H` controls has `H + 1` positions.

Left and right refer to the route direction, not world coordinates. A pass trajectory must clear
the supplied static geometry with the robot footprint and configured margin. Static clearance
does not establish safety around pedestrians; prediction and dynamic-risk arbitration belong to
separate components.

Missing geometry is not a known-empty scene: passing requires an explicit static check. Passing
can also be unavailable when the current motion cannot realize the requested route-relative side
within the horizon. Inspect rejection reasons rather than assuming every call produces every
family.

Braking is a trajectory, not an instantaneous zero-velocity command. The report distinguishes
stopping within the horizon from braking that has not yet reached rest. A candidate-count cap
must not remove a valid controlled-stop candidate. A statically infeasible stop remains a rejected
attempt, not a safety guarantee.

The generator reuses the existing route owner in `robot_sf.nav.global_route`, robot dynamics in
`robot_sf.robot.dynamics`, and risk input in
`robot_sf.research.collision_risk.estimators`. Existing planner defaults and benchmark outcomes
are unchanged. Numerical limits are configuration inputs, not measured hardware guarantees.

## Validation scope

The focused tests exercise deterministic generation, route-relative symmetry, actuator bounds,
static rejection, stopping, candidate caps, invalid inputs, immutable outputs, and compatibility
with the existing risk estimator. This is implementation evidence only: it does not establish
closed-loop planner benefit, calibrated risk, or collision avoidance.

See [the glossary](glossary.md) for project terminology and
[actuator feasibility](actuator_feasibility.md) for the distinction between clear space and
physically executable motion.

The current rollout uses velocity-controlled unicycle kinematics and assumes zero acceleration
immediately before the first command. Its jerk checks cover that explicit starting assumption;
they do not establish continuity with an unknown previous acceleration. The generator is an
opt-in component, not a hardware controller or an execution safety guard.

The canonical actuator feasibility evaluator includes configured command and brake latency in its
stopping-distance diagnostic. The unicycle rollout itself applies commands immediately; it does
not simulate delayed actuation. A delayed execution backend needs its own matching rollout and
validation before these trajectories can be treated as executable on that backend.
