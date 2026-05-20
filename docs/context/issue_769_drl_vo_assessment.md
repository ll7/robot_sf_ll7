# Issue #769 DRL-VO Assessment Note

This note documents the work completed for issue #769, which assessed and integrated `drl_vo` into Robot SF's benchmark metadata and readiness contracts.

> Status update, 2026-05-20: issue #1364 supersedes the original "no runtime planner" statement
> below. Robot SF now has a DRL-VO adapter, but the benchmark verdict remains experimental and
> opt-in rather than main-table ready.

## Summary

* Added `drl_vo` as a new benchmark candidate in `robot_sf/benchmark/algorithm_readiness.py`.
* Added `drl_vo` metadata support in `robot_sf/benchmark/algorithm_metadata.py`.
* Added a contract test in `tests/benchmark/test_algorithm_metadata_contract.py` to verify `drl_vo` metadata enrichment.
* Verified that the new metadata contract is accepted and that the code changes pass lint checks.

## What changed

1. `robot_sf/benchmark/algorithm_readiness.py`
   - Registered `drl_vo` in the `AlgorithmReadiness` table.
   - Marked it as experimental and opt-in, consistent with other nascent learning-based planners.

2. `robot_sf/benchmark/algorithm_metadata.py`
   - Extended the algorithm metadata contract to recognize `drl_vo` .
   - Provided baseline classification:

     - `baseline_category: learning`

     - `policy_semantics: drl_vo`

   - Added `upstream_reference` metadata for provenance.
   - Included a compatible `planner_kinematics` profile for the algorithm.

3. `tests/benchmark/test_algorithm_metadata_contract.py`
   - Added a test ensuring `enrich_algorithm_metadata("drl_vo", {...})` returns the expected canonical metadata fields.

## Verification

* Ran targeted unit tests for algorithm metadata contracts.
* Confirmed the new `drl_vo` test passes.
* Ran Ruff lint checks on the modified files.

Result:

* `pytest` for the benchmark metadata contract test passed.
* `uv run ruff check robot_sf/benchmark/algorithm_metadata.py robot_sf/benchmark/algorithm_readiness.py tests/benchmark/test_algorithm_metadata_contract.py` passed.

## Current result

* `drl_vo` is now represented in the benchmark metadata and readiness layer.
* The repository can recognize `drl_vo` as an experimental planner candidate for future benchmark integration.
* Historical note from #769: there was no runtime planner in the codebase at that time.
  That statement is superseded by the #1364 audit below.

## Issue #1364 Privileged-State Audit Update 2026-05-20

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1364>

### Current adapter surface

Robot SF now includes a concrete DRL-VO adapter surface:

* `robot_sf/baselines/drl_vo.py`
  * `DrlVoPlannerConfig`: `model_path`, `model_id`, `device`, `deterministic`, `nearest_k`,
    `action_space`, `v_max`, `omega_max`, `fallback_to_goal`, and `goal_speed`.
  * `DrlVoPlanner._build_model_input(...)`: flattens relative goal, robot velocity, and the nearest
    `nearest_k` pedestrian position/velocity rows.
  * `DrlVoPlanner._parse_model_action(...)`: accepts `{"vx", "vy"}` velocity actions or
    `{"v", "omega"}` unicycle actions, plus length-2 vector outputs.
  * `DrlVoPlanner._goal_seeking_action(...)`: deterministic fallback when no compatible model is
    available and `fallback_to_goal=true`.
* `robot_sf/benchmark/map_runner.py`
  * `_obs_to_ppo_format(...)` converts map-runner observations into the DRL-VO/PPO-style dict.
  * The `algo_key == "drl_vo"` branch calls `DrlVoPlanner.step(...)`, converts velocity-vector
    actions to Robot SF `unicycle_vw`, and records adapter-impact/feasibility metadata.
* `configs/baselines/drl_vo_default.yaml`
  * Provides the default opt-in config, with `allow_testing_algorithms: true` and
    `fallback_to_goal: true`.

### Field classification

The current Robot SF adapter consumes these fields:

| Field | Source path | Classification | Notes |
| --- | --- | --- | --- |
| `robot.position` | map-runner observation | deployment-observable | Robot self-state. |
| `robot.velocity` | map-runner observation | deployment-observable | Falls back from speed/heading in `_obs_to_ppo_format`. |
| `robot.heading` | map-runner observation | deployment-observable | Required for unicycle fallback/projection. |
| `robot.radius` | map-runner observation | deployment-observable | Passed through in the adapter payload. |
| `goal.current` | map-runner observation | deployment-observable | Converted to `robot.goal`. |
| `pedestrians.positions` | map-runner observation | benchmark-observable perfect tracks when the run declares `tracked_agents_no_noise`; oracle-only if the run declares `oracle_full_state` | Sorted by distance and capped to `nearest_k`; not a real-sensor certification. |
| `pedestrians.velocities` | map-runner observation | benchmark-observable perfect tracks when the run declares `tracked_agents_no_noise`; oracle-only if the run declares `oracle_full_state` | Missing rows are padded with zeros; not a real-sensor certification. |
| `pedestrians.count` | map-runner observation | deployment-observable count/mask metadata | Caps active rows before sorting/cropping. |
| `pedestrians.radius` | map-runner observation | deployment-observable | Shared radius passed to each agent row. |
| `sim.timestep` / `dt` | map-runner observation | deployment-observable | Used as adapter payload timing metadata. |
| `critic_privileged_state` | asymmetric critic observation only | forbidden for DRL-VO deployment | Not consumed by the DRL-VO adapter. |
| future pedestrian states, future ego trajectory, termination reason, metrics | none | forbidden | Not consumed by the current adapter path. |

### Source-side provenance boundary

The upstream DRL-VO paper describes a policy input made from short lidar history, nearby-pedestrian
kinematic data, and a subgoal point, with steering and forward-velocity output. The upstream
`TempleRAIL/drl_vo_nav` repository is GPL-3.0 and documents that its open-source training/testing
path uses accurate Gazebo pedestrian information instead of the detector/tracker pipeline used in
the paper, because the tracking stack could not be released under the same terms.

This matters for Robot SF: accurate simulator pedestrian state is acceptable only when reported as
the declared observation-level assumption. Under the current Robot SF metadata resolver, DRL-VO's
`socnav_state` default maps to `tracked_agents_no_noise`, which is a perfect-tracking benchmark
assumption, not real sensor certification. A run that instead declares `oracle_full_state` must be
reported as oracle/diagnostic evidence rather than fair main-table learned-policy evidence.

Sources:

* <https://github.com/TempleRAIL/drl_vo_nav>
* <https://arxiv.org/abs/2301.06512>

### Verdict

Verdict: `prototype-only / tracked-agent diagnostic`, not main-table ready.

The inspected adapter path is low risk for hidden privileged-state leakage because the local code
and tests route only Robot SF robot, goal, pedestrian, and timestep fields into the planner and do
not read `critic_privileged_state`, future trajectories, termination labels, or aggregate metrics.
That is adapter-contract evidence, not loaded-checkpoint benchmark proof. The adapter can support
fair diagnostic smokes when the run metadata declares `tracked_agents_no_noise` or a stricter
synthetic-noise level.

It is not main-table eligible yet because:

* the default config points at `output/model_cache/drl_vo_default.pt`, which is not a durable
  tracked checkpoint source;
* `fallback_to_goal=true` allows a missing or invalid model to run a goal-seeking fallback, which
  must be reported as fallback/degraded rather than DRL-VO evidence;
* no current Robot SF benchmark artifact proves a loaded DRL-VO checkpoint through the actual
  adapter path;
* upstream source reproduction still depends on a ROS/Gazebo/Turtlebot stack and documented
  accurate-pedestrian simulation inputs.

### Validation

Validated on 2026-05-20:

```bash
uv run pytest tests/planner/test_drl_vo.py tests/test_map_runner_ppo.py::test_build_policy_drl_vo_adapter_impact_updates_metadata tests/test_map_runner_ppo.py::test_obs_to_ppo_format_preserves_heading_for_unicycle_fallback tests/benchmark/test_algorithm_metadata_contract.py::test_drl_vo_metadata_exposes_reference_contract -q
```

Result: 18 passed. The targeted tests cover registration, fallback behavior, nearest-agent
sorting, action parsing, map-runner conversion/projection, and upstream metadata.

```bash
rg -n "drl_vo|DRL-VO|privileged|tracked_agents_no_noise|fallback_to_goal" docs/context/issue_769_drl_vo_assessment.md robot_sf/baselines/drl_vo.py robot_sf/benchmark/map_runner.py robot_sf/benchmark/algorithm_metadata.py tests/planner/test_drl_vo.py
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```

Result: the documented verdict and adapter field paths are discoverable; docs/proof consistency
passed for 3 changed files.

## Follow-up

* Provide a durable checkpoint/model manifest before treating DRL-VO as loaded-model evidence.
* Add or select a smoke config that fails closed when the model is missing instead of reporting
  fallback-to-goal as DRL-VO success.
* If maintainers want main-table learned-policy consideration, run a tracked-agent-level smoke or
  benchmark slice with explicit observation-level metadata and no fallback/degraded status.
