# Issue 605 gym-collision-avoidance Reference Note

Date: 2026-03-19
Related issues:
- `robot_sf_ll7#605` Assess gym-collision-avoidance as RVO/CADRL benchmark reference
- `robot_sf_ll7#624` planner quality audit workflow
- `robot_sf_ll7#632` Python-RVO2 benchmark integration prototype

## Why this repository matters now

The merged planner audit showed that the current local `sacadrl` result is not credible as a
paper-facing representative of the CADRL family, while the `orca` path is already covered through a
cleaner upstream-backed `Python-RVO2` prototype. That leaves one remaining question for the
benchmark roadmap:

- should `gym-collision-avoidance` be treated as the canonical external reference for the
  SA-CADRL / GA3C-CADRL family, and is it practical enough to justify a future reproduction spike?

This note answers that question conservatively.

## Canonical source assets

Use these upstream files as the canonical source anchors for the repository and its expected
contracts.

### Repository and runnable entrypoints

- [README](https://github.com/mit-acl/gym-collision-avoidance/blob/master/README.md)
- [setup.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/setup.py)
- [example.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/gym_collision_avoidance/experiments/src/example.py)
- [test_collision_avoidance.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/gym_collision_avoidance/tests/test_collision_avoidance.py)

### Environment and config contract

- [collision_avoidance_env.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/gym_collision_avoidance/envs/collision_avoidance_env.py)
- [config.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/gym_collision_avoidance/envs/config.py)
- [test_cases.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/gym_collision_avoidance/envs/test_cases.py)
- [agent.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/gym_collision_avoidance/envs/agent.py)

### Policy-family anchors

- [CADRLPolicy.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/gym_collision_avoidance/envs/policies/CADRLPolicy.py)
- [GA3CCADRLPolicy.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/gym_collision_avoidance/envs/policies/GA3CCADRLPolicy.py)
- [RVOPolicy.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/gym_collision_avoidance/envs/policies/RVOPolicy.py)

## What the source actually provides

Observed evidence from the checked-out upstream source:

- license: MIT
- environment stack: legacy `gym`, not `gymnasium`
- packaging/runtime: Python package with example and test entrypoints
- learned-policy assets: bundled SA-CADRL pickle files and GA3C-CADRL TensorFlow checkpoints
- baseline breadth: CADRL, GA3C-CADRL, PPO-CADRL, RVO, static, non-cooperative, external-policy hooks
- dynamics model: unicycle-style agents with explicit heading and turn-rate handling

Important practical implication:

- this repository is more runnable and benchmark-complete than a paper-only reference,
- but it is materially more legacy-bound than the current `robot_sf_ll7` stack.

## Integration shape decision

Decision:

- historical family anchor: `gym-collision-avoidance`
- strongest near-term use: `source-harness reproduction first`
- integration category: `prototype only`
- preferred future shape: `source-harness reproduction`, then `model-wrapping adapter`
- fallback policy: `fail fast only`

Rationale:

- it is the most concrete public reference for the SA-CADRL / GA3C-CADRL family because it ships
  runnable environment code, policy entrypoints, and bundled checkpoint assets,
- but the stack is Gym/TensorFlow-era and should not be forced directly into the main
  `robot_sf_ll7` runtime without first proving the source harness still runs cleanly,
- the existing upstream-backed ORCA route is already better served by `Python-RVO2`, so the main
  new value here is the CADRL-family reproduction path, not another RVO import.

## Observation and action contract translation

| Contract area | Source expectation | Robot SF supply/target | Judgment |
|---|---|---|---|
| observation | per-agent structured state dict keyed by configured state names such as `dist_to_goal`, `heading_ego_frame`, `pref_speed`, `radius`, and `other_agents_states`; optional laser scan modes also exist | Robot SF structured planner observations and benchmark metadata | direct compatibility: partial only |
| observation translation | source policies depend on exact state ordering, normalization, nearby-agent clipping, and source sensor semantics | Robot SF would need an explicit adapter that reconstructs the expected source state vector/dict exactly | adapter required: yes |
| action | source env uses continuous `[speed, delta_heading]` commands for unicycle-style agents; GA3C-CADRL can also emit discrete action choices internally before env translation | Robot SF benchmark expects `unicycle_vw` | direct compatibility: partial only |
| action translation | source commands are close to benchmark kinematics but not identical in parameterization | Robot SF would need an explicit `speed + delta_heading -> unicycle_vw` mapping, likely using `omega = delta_heading / dt` with the source `dt` kept explicit | post-policy adapter required: yes |

Interpretation:

- this family is much closer to Robot SF kinematics than holonomic crowd-navigation repos,
- but it is still not a drop-in planner interface,
- and policy faithfulness depends on reproducing the source observation contract before any wrapper claim.

## Benchmark-readiness risks

| Risk area | Assessment | Implication |
|---|---|---|
| dependency maturity | the repo was updated for Python 3.10 and TensorFlow 2, but it is still built around legacy `gym` and older RL-era assumptions | source-harness reproduction is plausible, main-stack import is not low-friction |
| source-harness reproducibility | example and test entrypoints exist, which is stronger than many benchmark references | this is a good candidate for a fail-fast source-harness spike |
| model availability | bundled SA-CADRL and GA3C-CADRL checkpoints are present in-tree | learned-family reproduction is materially more plausible than with code-only repos |
| observation contract coupling | strong; CADRL/GA3C-CADRL rely on exact state packing and source normalization | wrapper work must remain explicit and should not start before source-harness validation |
| action / kinematics mismatch | moderate rather than fatal; source is already unicycle-like, but uses `[speed, delta_heading]` rather than benchmark `unicycle_vw` | adapter is feasible, but conversion semantics must be documented |
| benchmark credibility risk | medium if wrapped carefully, high if partially reimplemented | provenance must stay upstream-first and fail-fast |
| ORCA overlap | high overlap with the already stronger `Python-RVO2` ORCA path | future work should focus on CADRL-family value, not another ORCA wrapper |

## Final recommendation

Recommendation: `prototype only`

Interpretation boundary:

- treat `gym-collision-avoidance` as the canonical external reference for the SA-CADRL / GA3C-CADRL
  family,
- do not treat it as a drop-in import target for the current benchmark runtime,
- and do not spend effort on its RVO path as a primary benchmark addition because `Python-RVO2`
  already covers that baseline more cleanly.

Best next implementation shape if this family is selected later:

1. run the upstream example and one bundled learned-policy test path in an isolated side environment,
2. document the exact observation packing and action semantics from the source harness,
3. only then prototype a fail-fast Robot SF wrapper for one learned CADRL-family policy,
4. keep claims at implementation level until source-harness and wrapper parity are both documented.

Not recommended in this issue:

- direct runtime integration into the main `robot_sf_ll7` environment stack
- another ORCA-focused import path
- family-level benchmark claims without source-harness validation
