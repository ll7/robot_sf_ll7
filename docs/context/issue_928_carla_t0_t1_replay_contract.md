# Issue 928 CARLA T0/T1 Oracle Replay Contract

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/928>

Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/872>

Canonical roadmap: [docs/plan/plan_big_picture_2026-04-30.md](../plan/plan_big_picture_2026-04-30.md)

Scenario-certification predecessor:
[docs/context/issue_868_scenario_certification.md](issue_868_scenario_certification.md)

## Goal

Define the first CARLA transfer contract before adding optional simulator code. The first useful
milestone is oracle replay and parity for certified Robot-SF scenarios, not CARLA training,
perception stress, ROS integration, or paper-facing simulator-transfer claims.

This note scopes the parent issue into two early stages:

- T0: export certified Robot-SF scenario definitions to a neutral replay description without
  importing CARLA or requiring CARLA assets.
- T1: replay that neutral description in CARLA with oracle state, scripted pedestrians, and
  trajectory-level metrics when a CARLA runtime is explicitly available.

## Contract

T0 neutral export should be treated as a simulator-independent data contract. It should include the
minimum fields needed to reconstruct the Robot-SF scenario and compare trajectories later:

- scenario identity, source config, map identity, and certificate reference,
- robot start, goal, footprint, kinematic profile, and command/action interpretation,
- pedestrian initial states, scripted route or behavior metadata, and timing parameters,
- static obstacle and route/topology references or a durable exported geometry representation,
- time step, horizon, termination conditions, and metric configuration,
- provenance fields that identify the Robot-SF commit, config, and certificate generator version.

T1 replay should remain an optional integration path. It may import the CARLA Python API only behind
explicit dependency checks, and it should report `not available` when CARLA, a requested CARLA map,
or a certified scenario input is missing. Missing runtime support is not a benchmark failure by
itself; treating a fallback or partial replay as successful parity would be the failure.

## Required Behavior

- Normal Robot-SF installation, tests, training, and benchmark commands must not require CARLA.
- Bridge imports must fail closed with actionable errors when optional dependencies are missing.
- Scenario export must require a valid scenario certificate before it is used for transfer claims.
- CARLA replay outputs must identify mode as `oracle-replay`, `failed`, or `not-available`.
- Metric comparison should use trajectory-level fields first: success, collision, minimum distance,
  TTC where available, comfort, jerk, curvature, intervention rate, and SNQI-compatible fields when
  the input data supports them.
- Reports must clearly distinguish Robot-SF evidence from CARLA replay evidence.

## Out Of Scope

- CARLA policy training.
- Sensor-level perception stress tests.
- ROS or autonomous-driving stack integration.
- Claiming transfer or simulator parity from exported JSON alone.
- Counting degraded, fallback, or hand-authored partial replay as benchmark-strengthening evidence.
- Selecting the final first-ten certified scenario set before `scenario_cert.v1` coverage and
  counterexample replay bundles are stable enough to justify that selection.

## Fail-Closed Semantics

Future bridge commands should prefer explicit statuses over silent fallback:

- `not-available`: CARLA runtime, CARLA assets, optional Python package, or certified scenario input
  is unavailable.
- `failed`: replay was attempted but violated the contract, crashed, exceeded a timeout, or produced
  invalid/incomplete outputs.
- `oracle-replay`: CARLA replay ran with ground-truth state and scripted agents; this is parity
  evidence only when paired with comparable Robot-SF trajectory metrics.

Any `not-available` or `failed` status is valid diagnostic output, but it must not be reported as a
successful benchmark outcome.

## Validation Path

This issue is docs-only. Validation for the contract PR should prove discoverability and avoid
claiming runtime behavior:

```bash
git diff --check
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

Future implementation children should add CARLA-free tests for the neutral export schema and
dependency-guard tests for missing CARLA before requiring a CARLA-capable smoke environment.

## Follow-Up Boundary

The next implementation issue under #872 should add the T0 neutral export schema and a dependency
guard without importing CARLA in normal test collection. A later T1 issue can add a CARLA runtime
smoke command once a CARLA-capable machine and first certified scenario set are identified.
