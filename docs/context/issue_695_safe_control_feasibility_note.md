# Issue 695 `safe_control` Feasibility Note

Date: 2026-03-27
Related issues:
- `robot_sf_ll7#695` `safe_control` feasibility-first integration assessment
- `robot_sf_ll7#629` planner-zoo research prompt
- `robot_sf_ll7#601` CrowdNav family feasibility note
- `robot_sf_ll7#599` Go-MPC assessment note

## Goal

Assess whether [`tkkim-robot/safe_control`](https://github.com/tkkim-robot/safe_control) can be
integrated into `robot_sf_ll7` as a provenance-preserving, optional local-planner candidate without
vendoring upstream code, changing the default dependency set, or overstating benchmark support.

This issue is a feasibility/provenance issue first. It is **not** a paper-facing benchmark
integration claim.

## Canonical source anchors

- upstream repo: <https://github.com/tkkim-robot/safe_control>
- README: <https://github.com/tkkim-robot/safe_control/blob/main/README.md>
- packaging metadata: <https://github.com/tkkim-robot/safe_control/blob/main/pyproject.toml>
- tracking entrypoint: <https://github.com/tkkim-robot/safe_control/blob/main/tracking.py>
- example scripts:
  - <https://github.com/tkkim-robot/safe_control/blob/main/examples/test_tracking.py>
  - <https://github.com/tkkim-robot/safe_control/blob/main/examples/test_unknown_env.py>

## What the upstream project actually is

`safe_control` is best understood as a general safety-controller library rather than a benchmark
native local planner.

Observed upstream shape:

- It exposes a `LocalTrackingController` that follows waypoints while enforcing safety constraints.
- It supports many robot models, including `Unicycle2D`, `DynamicUnicycle2D`,
  `SingleIntegrator2D`, bicycle models, VTOL, and quadrotors.
- It organizes functionality around controller families such as `cbf_qp`, `mpc_cbf`,
  optimal-decay variants, and gatekeeper-style shielding.
- The library expects an upstream environment object and obstacle arrays rather than a Robot SF
  benchmark observation/action contract.

This means the natural integration shape is not:

- "drop-in local planner adapter"

It is closer to:

- "external tracking-controller runtime with an explicit state, waypoint, and obstacle bridge"

## Provenance and dependency assessment

### License and reuse boundary

Observed facts:

- GitHub does not currently report a detected license for the repository metadata.
- The upstream README and `pyproject.toml` are public, but this assessment did not confirm a clear
  SPDX-style license file through GitHub metadata.

Implication:

- direct vendoring is not justified
- any future integration must keep upstream code external and optional
- benchmark-family claims must stay conservative until license clarity is explicit

Decision:

- provenance boundary: **no vendoring**
- fallback policy: **fail fast only**

### Dependency burden

Upstream `pyproject.toml` declares:

- `numpy==1.26.4`
- `scipy`
- `matplotlib`
- `cvxpy`
- `casadi`
- `do-mpc[full]>=5.1.1`
- `shapely>=2.0.7`
- `gurobipy`

Local viability check in the current `robot_sf_ll7` environment:

- `cvxpy`: missing
- `casadi`: missing
- `do_mpc`: missing
- `gurobipy`: missing

Implication:

- the repo does **not** currently have the upstream runtime available
- even the smallest plausible controller path would require optional dependency installation before
  a truthful smoke run is possible
- the issue's narrow-spike gate therefore does not pass under the current environment

## Observation and action contract mismatch

| Contract area | `safe_control` expectation | `robot_sf_ll7` supply/target | Judgment |
| --- | --- | --- | --- |
| planner inputs | waypoint list, robot state, robot model spec, explicit obstacle arrays, optional upstream env object | benchmark-facing structured state or model-specific observations | direct compatibility: no |
| obstacle representation | circular and padded/superellipsoid obstacle arrays, optionally unknown obstacles with extra slots | map-derived obstacle segments and planner-facing structured obstacle summaries | adapter required: yes |
| robot state | model-specific state vector, with closest benchmark-aligned path being `Unicycle2D` | Robot SF robot pose/velocity/goal contract | adapter required: yes |
| action semantics | controller-generated low-level control consistent with the selected robot model | Robot SF benchmark expects `unicycle_vw` for paper-facing planners | partial compatibility only for the unicycle path |

Most important mismatch:

- `safe_control` wants a local tracking problem with waypoints and obstacle arrays
- `robot_sf_ll7` benchmark planners are evaluated as local-navigation policies with explicit
  benchmark contracts and scenario semantics

So even the best-case mapping is a narrow experimental bridge, not a family-level support claim.

## Viability-gate decision

The issue plan allowed a narrow spike only if all of these were true:

1. the package can stay external and optional
2. one controller path can run without changing the repo's default dependency set
3. one robot-model mapping is straightforward
4. one static-obstacle scenario can be exercised without overclaiming the result

Result:

- gate 1: **pass**
- gate 2: **fail**
- gate 3: **partial pass** for `Unicycle2D`
- gate 4: **pass in principle**, but only after gate 2 is resolved

Because gate 2 fails in the current environment, this issue should stop at the feasibility note.

## Integration-shape judgment

Preferred first family if work ever resumes:

- controller family: `cbf_qp`
- robot model: `Unicycle2D`
- scenario class: one deterministic static-obstacle smoke scenario

Recommended integration category:

- `prototype only`

Benchmark claim:

- `not supported yet`

Recommended future path if someone wants to continue:

1. confirm explicit upstream licensing/reuse status
2. reproduce one upstream `Unicycle2D` `cbf_qp` example in an isolated optional environment
3. only then decide whether a Robot SF wrapper spike is justified

## Final recommendation

Recommendation: **do not pursue now**

Reason:

- license clarity is not yet strong enough for broad integration confidence
- the dependency burden is materially outside the current default environment
- the controller contract is waypoint-tracking and obstacle-array driven, not benchmark-native
- there is no defensible narrow in-repo spike today without first doing external runtime work

Safe claim boundary:

- `safe_control` is a useful external safety-controller reference
- it is **not** current in-repo benchmark support
- it is **not** a justified paper-facing planner-family addition

## What would change this verdict

This verdict should be revisited only if all of the following become true:

- explicit upstream licensing is confirmed
- a minimal optional environment can run `Unicycle2D` + `cbf_qp`
- one static-obstacle source-style example is reproducible
- the resulting control output can be mapped to Robot SF `unicycle_vw` without semantic guessing
