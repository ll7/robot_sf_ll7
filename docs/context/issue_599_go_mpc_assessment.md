# Issue 599 Go-MPC Assessment Note

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#599` Go-MPC feasibility spike
- `robot_sf_ll7#593` predictive planner v2 / staged benchmark comparison
- `robot_sf_ll7#592` hybrid graph + obstacle-context predictive model idea
- `robot_sf_ll7#603` planner-family coverage matrix

## Goal

Assess Go-MPC as an external prediction-based planner-family anchor against the current predictive
planner stack in `robot_sf_ll7`.

This issue is an external-family feasibility and overlap assessment. It is **not** an implementation
task.

## Canonical source anchors

- upstream repo: <https://github.com/tud-amr/go-mpc>
- local checkout: `output/repos/go-mpc`
- paper: *Where to go next: Learning a Subgoal Recommendation Policy for Navigation Among Pedestrians*
- key entrypoints:
  - `output/repos/go-mpc/test.py`
  - `output/repos/go-mpc/test.sh`
  - `output/repos/go-mpc/mpc_rl_collision_avoidance/policies/MPCPolicy.py`
- license:
  - `output/repos/go-mpc/LICENSE`

## What the upstream method actually is

Go-MPC is not a simple local policy adapter.

It is a hybrid stack with two coupled parts:

1. a learned RL policy that recommends a subgoal or guidance target,
2. a FORCESPro-backed kinodynamic MPC controller that solves the local motion problem.

The local MPC policy path in `MPCPolicy.py` makes this explicit:

- the robot state is `[x, y, theta, v, w]`,
- nearby agents are injected as obstacle parameters,
- the controller calls `FORCESNLPsolver_py.FORCESNLPsolver_solve(...)`,
- the returned policy action is effectively a waypoint delta while the real motion command is
  produced by the solver-side internal state update.

This matters because the benchmark-facing integration shape would not be:

- “adapter only”

It would be:

- “hybrid RL + proprietary MPC solver runtime reproduction”

## Runtime and licensing assessment

Observed blockers from the upstream assets:

- license is `GPL-3.0`
- the README explicitly requires:
  - a separate `gym-collision-avoidance` clone,
  - Stable Baselines 2,
  - MATLAB + FORCESPro to generate the MPC solver,
  - a generated solver under `mpc_rl_collision_avoidance/mpc`
- the checked-in controller imports:
  - `from mpc_rl_collision_avoidance.mpc import FORCESNLPsolver_py`

Implication:

- even source-harness reproduction is not lightweight,
- faithful reuse is solver-locked,
- direct code import is a bad fit for a permissive, Gymnasium-first benchmark repo.

## Overlap with current internal planners

### Relative to `prediction_planner`

Current in-repo role:

- `prediction_planner` is a native Robot SF predictive local planner with learned short-horizon
  pedestrian forecasting and explicit rollout scoring.

Overlap with Go-MPC:

- both are prediction-aware local planners,
- both try to trade off goal progress against crowd interaction risk,
- both rely on short-horizon reasoning rather than pure reactive avoidance.

Key difference:

- `prediction_planner` is a direct benchmark-native local planner,
- Go-MPC is a two-level architecture where RL proposes a subgoal and MPC executes the control law.

### Relative to `predictive_mppi`

Current in-repo role:

- `predictive_mppi` is a native short-horizon sequence optimizer over learned pedestrian forecasts.

Overlap with Go-MPC:

- short-horizon optimization,
- prediction-conditioned scoring,
- explicit control-sequence search under robot dynamics.

Key difference:

- `predictive_mppi` stays entirely in-repo and emits explicit unicycle controls,
- Go-MPC depends on an external solver and an RL-generated guidance layer.

## Required inputs and action semantics

From the upstream code path:

- required obstacle inputs:
  - other-agent global positions,
  - other-agent global velocities,
  - combined obstacle radii / distance ordering
- required robot state:
  - global pose `(x, y, theta)`,
  - current linear and angular velocity `(v, w)`
- planner action semantics:
  - the RL network output is not the final robot command,
  - the MPC policy turns that guidance into a solver-generated kinodynamic plan,
  - the exposed action returned to the environment is effectively a next-waypoint delta

Kinematics judgment:

- Go-MPC is closer to differential-drive / kinodynamic execution than ORCA-family holonomic methods,
- but that benefit comes bundled with the solver/runtime burden above.

## Integration-shape judgment

Required integration shape for faithful reuse:

- broader runtime contract change

Why:

- adapter-only reuse would erase the real method identity,
- the method depends on:
  - RL guidance,
  - MPC solver generation,
  - legacy environment/runtime coupling,
  - solver-generated binaries or code.

It is therefore not comparable to:

- wrapping `Python-RVO2`,
- wrapping `Social-Navigation-PyEnvs` non-trainable policies,
- or running a model-only checkpoint probe.

## Recommendation

Recommendation: `do not pursue now`

Reason:

- GPL-3.0 is a poor fit for direct vendoring,
- FORCESPro solver generation is a hard operational blocker,
- the runtime stack is legacy and heavyweight,
- the benchmark already has native predictive planners whose next improvements are cheaper and more
  controllable than faithful Go-MPC reproduction.

## What remains valuable

Go-MPC is still useful as a design reference.

Reusable ideas worth keeping in scope:

- learned subgoal recommendation layered over a lower-level optimizer,
- prediction-conditioned guidance rather than raw reactive control,
- separating strategic crowd-aware guidance from local kinodynamic execution.

Those ideas already map more naturally onto:

- `robot_sf_ll7#593` prediction planner v2,
- `robot_sf_ll7#592` hybrid guidance / obstacle-context ideas,

than onto a direct external-code integration effort.
