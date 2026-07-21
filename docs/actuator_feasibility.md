# Actuator-Feasibility Validation (Issue #6056)

> **Status: experimental diagnostic.** This layer is **not** a formal safety case, it is
> not conformalized, it is not learned, and it does **not** change default planner
> behavior. It is a pure, deterministic, side-effect-free evaluator that can be run
> offline or in tests. See the claim boundary at the end of this document.

## Plain-language summary

A planned maneuver is only useful if the robot can actually execute it. A path that
*looks* clear on the map can still be impossible to follow: the robot may not be able to
brake fast enough, turn sharply enough, change steering quickly enough, or overcome
command and brake latency in time. This module separates two questions that existing
trajectory-verifier and clearance work treat together:

- **Geometrically clear** — is there *any* room at all? Has the robot already reached the
  hazard? This is a pure geometry question (clearance ≥ 0).
- **Physically feasible** — can the robot execute the maneuver given its acceleration,
  braking, yaw-rate, steering-rate, and latency limits?

The value of the layer is the **distinction** between the two. A maneuver can be
geometrically clear (there is room) yet physically infeasible (the robot cannot stop,
turn, or steer fast enough, or latency eats the available room). The evaluator reports an
explicit verdict and lists exactly which actuator limit was violated.

AMMV stands for *Autonomous Micromobility Vehicle* — the three-wheeled target platform for
which this work is a deployment-critical layer.

## ⚠️ Provisional values

**All numeric actuator limits below are provisional defaults unless derived from a
measured target platform (e.g. a measured AMMV platform).** They are conservative round
numbers chosen so the evaluator is usable out of the box; they are **not** hardware
claims and must not be cited as measured platform limits. Before using these checks for
anything stronger than a diagnostic, replace the defaults with values measured on the
actual platform and record their provenance.

## Where the code lives

- Evaluator and config schema: `robot_sf/benchmark/actuator_feasibility.py`
- Tests: `tests/benchmark/test_actuator_feasibility.py`
- Diagnostic smoke: `scripts/validation/run_actuator_feasibility_smoke.py`

The evaluator is intentionally self-contained and is **not** wired into any planner
control loop, release gate, or benchmark scoring by default. Wiring it in is an explicit,
separate decision.

## Config schema

The actuator limits are represented by `ActuatorLimitsConfig`, a frozen dataclass, and can
be loaded from a config mapping via `load_actuator_limits`. The schema version is
`actuator_feasibility.v1`. The block may be supplied nested under `actuator_limits` or
flat; an optional `schema_version` field, when present, must equal `actuator_feasibility.v1`.

```yaml
actuator_limits:
  schema_version: actuator_feasibility.v1
  max_accel_mps2: 1.0          # max forward acceleration magnitude (m/s^2)
  max_decel_mps2: 1.5          # max braking deceleration magnitude (m/s^2)
  max_yaw_rate_radps: 1.0      # max yaw-rate magnitude (rad/s)
  max_steering_rate_radps: 0.5 # max rate of change of yaw rate (rad/s^2), steering proxy
  command_latency_s: 0.15      # command/reaction latency before a new command acts (s)
  brake_latency_s: 0.2         # brake-engagement latency before deceleration begins (s)
```

The four required categories from the issue are all represented: **acceleration**
(`max_accel_mps2`), **braking** (`max_decel_mps2`), **yaw/steering rate**
(`max_yaw_rate_radps`, `max_steering_rate_radps`), and **latency** (`command_latency_s`,
`brake_latency_s`). The loader fails closed: a malformed block or wrong schema version
raises `ValueError` rather than silently weakening the limits. Unknown or misspelled keys
also raise `ValueError`; omitted supported fields use the provisional defaults shown above.

## Checks

The evaluator computes a per-trajectory set of observed maxima and compares each against
the configured authority. The fallback-brake deadline is evaluated from the conservative
worst-case (maximum) speed over the trajectory.

| Predicate id | What it checks |
| --- | --- |
| `accel_limit_exceeded` | Commanded forward acceleration magnitude exceeds `max_accel_mps2`. |
| `decel_limit_exceeded` | Commanded braking deceleration magnitude exceeds `max_decel_mps2`. |
| `yaw_rate_limit_exceeded` | Commanded yaw-rate magnitude exceeds `max_yaw_rate_radps`. |
| `steering_rate_limit_exceeded` | Rate of change of yaw rate exceeds `max_steering_rate_radps` (a steering-discontinuity proxy). |
| `fallback_brake_deadline_missed` | Latency-inclusive stopping distance exceeds the available geometric clearance. |

The latency-inclusive stopping distance is

```
d_stop = v^2 / (2 * max_decel_mps2) + v * (command_latency_s + brake_latency_s)
```

so the fallback-brake deadline is missed when `d_stop > available_clearance_m`. The
latency term is the part a geometry-only check misses: two encounters with identical speed
and clearance can differ in feasibility purely because of latency.

## Verdicts

`evaluate_actuator_feasibility` returns an `ActuatorFeasibilityReport` whose `verdict`
combines the two questions:

| Verdict | Meaning |
| --- | --- |
| `actuator_feasible` | Geometrically clear **and** physically feasible. |
| `geometry_only_clear` | Geometrically clear **but** physically infeasible (the distinguishing case). |
| `infeasible` | Not geometrically clear — the robot has already reached the hazard (clearance < 0), regardless of actuator limits. |

`violated_limits` lists exactly which actuator-limit predicate(s) fired, in evaluation
order (brake deadline first, then the per-trajectory rate checks).

## Example

```python
from robot_sf.benchmark.actuator_feasibility import (
    ActuatorLimitsConfig,
    evaluate_actuator_feasibility,
)

# 1 m/s straight approach; 0.5 m of geometric clearance ahead.
report = evaluate_actuator_feasibility(
    robot_positions=robot_positions,   # shape (T, 2)
    robot_velocities=robot_velocities, # shape (T, 2)
    dt_s=0.1,
    hazard_clearance_m=0.5,
    config=ActuatorLimitsConfig(),  # provisional defaults
)
# report.verdict           -> "geometry_only_clear"
# report.geometrically_clear -> True   (there is room)
# report.physically_feasible -> False  (cannot stop in time with latency)
# report.violated_limits   -> ("fallback_brake_deadline_missed",)
# report.stopping_distance_m -> ~0.68 m > 0.5 m clearance
```

For trajectories that can rotate in place, pass `robot_headings` (radians, shape `(T,)`) or
authoritative `robot_angular_velocities` (rad/s, shape `(T - 1,)` or `(T,)`). These signals
are required to check yaw and steering feasibility while translational velocity is zero;
without them, the evaluator falls back to yaw inferred from adjacent moving velocity
headings.

A companion entry point `evaluate_encounter_actuator_feasibility` evaluates a single
encounter from just the current speed and clearance (no trajectory needed); it applies the
brake-deadline check only.

## Diagnostic smoke

`scripts/validation/run_actuator_feasibility_smoke.py` runs the evaluator over a small,
deterministic scenario table and writes a JSON + Markdown report to
`output/diagnostics/actuator_feasibility_smoke_issue_6056/` showing every verdict and the
violated limit per row. It is diagnostic only.

```bash
uv run python scripts/validation/run_actuator_feasibility_smoke.py
```

## Related work

This complements, and does not replace, the experimental trajectory verifier
(`robot_sf/benchmark/trajectory_verifier.py`, issue #4757) and the footprint /
clearance-semantics diagnostic (`robot_sf/benchmark/clearance_semantics.py`, issue #3207).
The trajectory verifier already includes a braking-feasibility predicate using a proxy
`max_brake_deceleration_mps2`; this layer adds the explicit **latency**, **acceleration**,
**yaw-rate**, and **steering-rate** limits and the geometry-only vs physically-feasible
verdict split.

## Claim boundary

```
experimental actuator-feasibility diagnostic; not a formal safety case; not
conformalized; not learned; default planner behavior unchanged; numeric limits are
PROVISIONAL defaults unless derived from a measured target platform (e.g. AMMV);
geometrically-clear does not imply physically-feasible
```
