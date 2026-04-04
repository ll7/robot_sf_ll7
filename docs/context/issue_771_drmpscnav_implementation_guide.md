# Issue 771 MPC Social Navigation Implementation, Testing, and Validation Guide

## Goal

Establish a concrete Robot SF pathway for evaluating MPC-based social navigation candidates through implementation, testing, and benchmark metric validation until the selected policy has credible, repeatable evidence.

## Scope

* In scope: DR-MPC and SICNav candidate evaluation, adapter prototyping, and benchmark-safe validation.
* In scope: source/repo readiness, interface mapping, dependency/governance risk, and benchmark metric evidence.
* In scope: determining whether one or both methods remain `prototype only`,  `assessment only`, or `do not pursue now`.
* Out of scope: training a new controller from scratch as part of this spike, and treating fallback execution as a successful benchmark outcome.

## Outcome

This guide is complete when the selected policy has:
1. a minimal Robot SF integration design,
2. a smoke-tested wrapper on the verified-simple gate,
3. repeatable benchmark metrics on a controlled scenario subset,
4. a clear promotion or deferral recommendation.

## Candidate summary

### DR-MPC

* Status: `assessment only`.
* Upstream repo: available, but no obvious published pretrained model artifact in repo root.
* Interface: residual action on top of a path-tracking MPC base command, requiring rich robot and neighbor history state.
* Primary risk: missing license file, non-robot-SF simulation stack, and checkpoint uncertainty.

### SICNav

* Status: `prototype only` for benchmark evaluation.
* Upstream repo: MIT license, checkpoint artifacts available.
* Interface: bilevel MPC policy selection via `sicnav` / `sicnav_diffusion` classes.
* Primary risk: solver-heavy dependency stack (`CasADi`,  `IPOPT`/`acados`), runtime complexity, and environment mapping cost.

## Implementation plan

### 1. Establish the benchmark-safe boundary

* Keep new MPC adapters behind explicit protective flags:
  + `allow_testing_algorithms: true`
  + `include_in_paper: false`
* Add external anchor rows for `DR-MPC` and `SICNav` to `docs/benchmark_planner_family_coverage.md` as `conceptually adjacent only`.
* Implement only a thin wrapper that maps Robot SF state/action into the upstream candidate API.

### 2. Prioritize the first candidate

* Use `SICNav` as the first new policy to validate because it has cleaner licensing and checkpoint evidence.
* Retain `DR-MPC` as an assessment-case anchor until a packaged checkpoint and license clearance exist.

### 3. Build the thin adapter

The wrapper must expose the candidate as a Robot SF planner without changing the upstream core algorithm.

#### Robot state translation

* pose and heading `(x, y, theta)`
* linear/angular velocity or equivalent motion state
* goal position and path or waypoint reference
* robot collision geometry

#### Human state translation

* each pedestrian's position, velocity, and radius
* history or prediction input required by the candidate
* crowd state encoding consistent with upstream format

#### Action back-projection

* map upstream output to Robot SF command space:
  + `unicycle_vw` for 2D velocity/turn rate commands, 
  + if needed, curvature/velocity conversion for `SICNav`.
* maintain semantics of safe interaction decisions rather than post-hoc clipping.

### 4. Dependency and provenance guardrails

* Record upstream repo URL, checkpoint path, and license status in the implementation note.
* Use explicit `SOURCE` provenance if the candidate is vendor-copied or wrapped externally.
* Fail closed in CI if required upstream dependencies are unavailable.
* Prefer `SICNav` solver configuration that matches the available checkpoint and the runner used by upstream examples.

## Testing plan

### 1. Smoke execution on the verified-simple gate

* Validate the candidate on `configs/scenarios/sets/verified_simple_subset_v1.yaml`.
* Use a config with:
  + `allow_testing_algorithms: true`
  + `include_in_paper: false`
  + `algo=sicnav` for the initial prototype
* Confirm the wrapper:
  + starts cleanly, 
  + executes episodes without crashing, 
  + produces valid Robot SF control actions, 
  + does not silently fall back to a different planner.

### 2. Representative scenario coverage

* Cover at least one static obstacle scenario and one dynamic pedestrian interaction scenario from the verified-simple subset.
* Recommended cases:
  + `narrow_passage` or `single_obstacle_circle`
  + `single_ped_crossing_orthogonal` or `head_on_interaction`

### 3. Solver/runtime validation

* Measure per-step inference/solve time.
* Verify the policy does not violate the environment step budget.
* Confirm the chosen solver configuration is correctly installed and invoked.
* If a solver is unavailable, the policy should fail fast with a clear message.

### 4. Repeatability check

* Run the wrapper with at least three different random seeds.
* Compare the summary metrics for large variance or solver instability.
* If the candidate is unstable, retain `prototype only` and document the instability.

## Benchmark validation guide

### 1. Required metrics

Validate the policy using the benchmark metrics that matter for Robot SF comparisons:
* success / goal completion rate
* collision count or collision rate
* near-miss / social interaction metric where available
* path efficiency or progress measure
* runtime cost and solver performance

### 2. Validation stages

* Stage 1: verified-simple gate
  + first pass for viability and data sanity
* Stage 2: broader atomic suite
  + only after Stage 1 passes with stable, repeatable metrics
* Stage 3: documented evidence
  + compare candidate metrics to existing baseline policies and experimental anchors

### 3. Promotion criteria

Keep the candidate behind the testing guard until it satisfies all of the following:
* repeatable benchmark evidence on the verified-simple suite, 
* contradiction-free outputs (no solver crashes, invalid actions, or pathing exploration failures), 
* stable behavior across repeated seeds, 
* a documented case for why the candidate should exist alongside current baselines.

If Stage 1 fails, do not escalate the candidate to broader benchmark runs. If Stage 1 passes but metrics are weak, keep the candidate as `prototype only` and capture the failure mode.

### 4. Failure and deferral documentation

If the new policy does not validate, record:
* why the policy failed (installation, solver instability, zero success, collisions, runtime blow-up), 
* whether the issue is implementation-mapped or algorithm-intrinsic, 
* the current readiness classification (`assessment only` or `prototype only`).

## Governance and next steps

### 1. Keep the testing guard explicit

* Do not remove `allow_testing_algorithms: true` until the candidate has explicit benchmark promotion evidence.
* Keep `include_in_paper: false` until the policy is accepted as a stable benchmark participant.
* Keep the candidate outside routine camera-ready matrices until its performance is proven.

### 2. Document the decision path

* Use `docs/benchmark_planner_family_coverage.md` to anchor the current candidate status.
* Keep `docs/context/issue_771_drmpscnav_assessment.md` as the canonical assessment note.
* Store the implementation/testing evidence in a follow-up note or issue.

### 3. Create the wrapper implementation issue

Open a dedicated follow-up issue such as `#771-drmpscnav-implementation` that covers:
* implementation of the thin wrapper adapter, 
* CI smoke-run config for the verified-simple gate, 
* benchmark metric validation documentation.

## Recommended path

* Proceed with `SICNav` as the first new policy candidate for implementation and benchmark validation.
* Keep `DR-MPC` as an upstream assessment anchor until a packaged checkpoint and license status are resolved.
* Do not treat solver fallback or partial runtime success as validated benchmark evidence.

## Validation completion signal

This guide is fulfilled when the new policy has:
* been executed successfully on the verified-simple benchmark gate, 
* produced repeatable benchmark metrics, 
* been compared to baseline policy outcomes, 
* been documented with a clear promotion or deferral recommendation.
