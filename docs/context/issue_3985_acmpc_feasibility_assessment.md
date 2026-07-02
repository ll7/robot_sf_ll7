# Issue #3985 ACMPC-Inspired Learned-MPC Planner Feasibility Assessment

Issue: [#3985](https://github.com/ll7/robot_sf_ll7/issues/3985)
Date: 2026-07-02
Status: assessment-only. This note does not add a Robot SF planner, benchmark config, training job,
checkpoint, or paper-facing claim.

## Summary

Actor-Critic Model Predictive Control (ACMPC) is worth keeping as a concept source for a future
Robot SF-native learned model predictive control (MPC) planner, but it is not ready for direct
planner integration. The next useful step is a bounded design child issue, not source-code reuse or
benchmark execution.

Recommended verdict: **conditional go for a design-only child issue**, **no-go for immediate
implementation**.

Rough effort band:

- Design child issue: 1-2 focused days to specify dynamics, observation/action metadata, and a
  fail-closed smoke target.
- First runnable Robot SF prototype: 2-4 weeks if the differentiable MPC dependency and training
  loop are accepted; longer if the Stable-Baselines3 fork or custom differentiable dynamics become
  invasive.

## Sources Checked

- ACMPC upstream repository: <https://github.com/uzh-rpg/acmpc_public>
- ACMPC paper: <https://arxiv.org/abs/2306.09852>
- ACMPC differentiable MPC dependency fork: <https://github.com/uzh-rpg/mpc.pytorch_acmpc>
- Robot SF planner contribution contract: [`docs/contributing_planner.md`](../contributing_planner.md)
- Learned local-policy adapter contract:
  [`docs/context/issue_1618_learned_policy_adapter_interface.md`](issue_1618_learned_policy_adapter_interface.md)
- Learned local-policy eligibility checklist:
  [`docs/context/policy_search/contracts/learned_local_policy_eligibility.md`](policy_search/contracts/learned_local_policy_eligibility.md)
- Planner family coverage matrix:
  [`docs/benchmark_planner_family_coverage.md`](../benchmark_planner_family_coverage.md)
- Map-runner policy-builder surfaces:
  [`robot_sf/benchmark/map_runner.py`](../../robot_sf/benchmark/map_runner.py) and
  [`robot_sf/benchmark/map_runner_policies/`](../../robot_sf/benchmark/map_runner_policies/)

## Upstream Method Boundary

ACMPC embeds a differentiable MPC layer inside an actor-critic reinforcement learning (RL) policy.
The source repository is an agile quadrotor racing implementation, not a social-navigation planner.
The transferable idea is the architecture:

- short-horizon differentiable MPC as the actor's final decision layer;
- learned critic value shaping MPC costs or terminal value;
- end-to-end training through the MPC layer;
- online replanning retained at inference time.

The non-transferable parts are the quadrotor dynamics, racing task rewards, drone-specific state
features, and reported flight performance. Those results are motivation only. They are not Robot SF
social-navigation evidence.

## Adapter-Burden Assessment

| Surface | Burden | Assessment |
| --- | --- | --- |
| Dynamics model | High | A faithful Robot SF path needs differentiable ground-robot dynamics, likely `unicycle_vw` first and optionally actuation-aware variants from the maneuver-authority lane. Reusing quadrotor dynamics would erase the planner identity for this domain. |
| Observation contract | High | A learned local planner must define `observation_t`, deployment-observable robot/goal/pedestrian/map fields, and forbidden future or outcome fields. This belongs under the learned-policy eligibility contract before any candidate registry entry. |
| Action contract | Medium-high | The likely output is a bounded velocity command or first action from a short MPC trajectory. Raw MPC output, clipped output, and post-guard command must be logged separately. |
| Differentiable MPC dependency | Medium-high | `mpc.pytorch_acmpc` is the relevant dependency anchor, but it should stay outside core dependencies until a source-harness or optional environment proves import, solve, and gradient behavior on a tiny Robot SF-style dynamics problem. |
| RL training integration | High | ACMPC changes the policy class and update path, not just the planner adapter. The upstream implementation relies on training modules and a Stable-Baselines3 fork, so a Robot SF prototype should avoid modifying benchmark runners first. |
| Runtime cost | Medium-high | Differentiable MPC inside the actor can be slow or numerically brittle. The first smoke should measure bounded inference latency and fail closed on solver failure instead of falling back to another planner as success. |
| Artifact provenance | High | Any learned checkpoint, normalizer, or training trace would need durable model-registry or artifact-manifest entries before benchmark comparison. This issue has no such artifact. |

## Integration-Point Map

Do not start by editing the benchmark matrix or paper profiles. The safest integration ladder is:

1. Add a design child issue with a Robot SF-native candidate name, for example
   `acmpc_unicycle_local_v0`.
2. Specify a learned local-policy eligibility YAML or equivalent checklist answer covering
   observation, action, split, normalizer, checkpoint, device, deterministic inference, and fallback
   behavior.
3. Add a tiny differentiable dynamics source-harness under an optional validation path, not a
   planner registry row. The harness should prove one finite forward solve and one gradient through
   a `unicycle_vw` or actuation-aware dynamics step.
4. Only after the harness passes, implement a testing-only planner under `robot_sf/planner/` with a
   config-first path under `configs/algos/` or `configs/policy_search/candidates/`.
5. Wire the planner through the existing map-runner policy path:
   `robot_sf/benchmark/map_runner.py` and `robot_sf/benchmark/map_runner_policies/`.
6. Add metadata in `robot_sf/benchmark/algorithm_metadata.py` and
   `robot_sf/benchmark/algorithm_readiness.py` as experimental, explicit-opt-in, and unavailable
   unless the required dependency and artifact contract are satisfied.
7. Add smoke tests for metadata, missing-dependency fail-closed behavior, command bounds, reset
   behavior, and per-step diagnostics before any benchmark run.

The first child should stop at steps 1-3 unless maintainers explicitly choose the larger prototype.

## Benchmark-Readiness Boundary

This assessment is `preflight` / `idea` evidence only.

Allowed statements:

- ACMPC is a relevant learned-MPC architecture reference.
- Direct reuse has high adapter burden because the released implementation targets quadrotor racing.
- A Robot SF-native design child is justified if the team wants a learned-MPC hybrid family distinct
  from existing predictive and residual-learning lanes.

Disallowed statements:

- ACMPC improves Robot SF social navigation.
- ACMPC is benchmark-ready, paper-ready, or represented by an in-repo planner.
- Drone-racing robustness or speed results transfer to pedestrian crowd navigation.
- Fallback, guard-mediated, missing-dependency, or degraded execution counts as planner success.

Paper-grade or benchmark-strength claims would require a runnable Robot SF planner, declared
observation/action contract, artifact provenance, fail-closed dependency behavior, smoke proof,
predeclared benchmark matrix, seeds, metrics, fallback/degraded exclusions, and documented
limitations.

## Positioning Against Adjacent Work

| Adjacent issue | Relationship | Distinction |
| --- | --- | --- |
| [#3953](https://github.com/ll7/robot_sf_ll7/issues/3953) prediction-aware MPC local planner | Classical or native prediction-aware MPC with predicted pedestrian futures and time-varying constraints. | ACMPC adds learned actor-critic training through a differentiable MPC policy layer. It should not replace the classical MPC route. |
| [#1358](https://github.com/ll7/robot_sf_ll7/issues/1358) ORCA-residual learned policy | Learned residual on top of a reactive ORCA-style baseline. | ACMPC would make the optimizer itself the learned policy layer, not just add a bounded residual to a fixed reactive command. |
| [#2835](https://github.com/ll7/robot_sf_ll7/issues/2835) forecast local-navigation lane | Parent research lane for forecast-aware local navigation. | ACMPC belongs as a possible learned-MPC hybrid family only if its dynamics and training contracts are made Robot SF-native. |
| [#3213](https://github.com/ll7/robot_sf_ll7/issues/3213) maneuver-authority / kinematic adapters | Provides relevant action-authority and kinematics evidence for future dynamics modeling. | ACMPC would depend on this style of dynamics contract but does not itself prove maneuver-authority gains. |

## Recommendation

Open a bounded child only if the maintainer wants to spend research capacity on learned-MPC
hybrids after the existing forecast and residual-learning lanes are considered. The child should be
framed as a **design and source-harness preflight**, not as planner implementation:

Proposed child objective:

> Define `acmpc_unicycle_local_v0` as a Robot SF-native learned-MPC design candidate and prove a
> tiny differentiable-MPC source harness over a declared `unicycle_vw` dynamics model. Do not add a
> benchmark planner row, training job, or checkpoint.

Minimum child acceptance criteria:

- observation and action contracts filled using the learned local-policy checklist;
- dependency decision for `mpc.pytorch_acmpc` or an alternative differentiable MPC layer recorded;
- one finite solve and gradient check on a tiny local dynamics fixture, or a fail-closed blocked
  verdict naming the dependency/dynamics blocker;
- explicit no-claim boundary for Robot SF benchmark performance.

If that child fails, keep ACMPC as a conceptually adjacent external anchor only.

## Validation

This PR is docs-only and assessment-only. No benchmark, Slurm job, GPU job, training run, or
planner execution was performed.
