# Issue #1614 LiDAR Planner Compatibility

Date: 2026-05-29

Related issue: https://github.com/ll7/robot_sf_ll7/issues/1614

## Goal

Classify representative planners for a future LiDAR-observation benchmark track without allowing
privileged simulator state, map grids, or direct pedestrian state to be counted as LiDAR-only
evidence.

This note is an audit and follow-up selector. It does not implement adapters, retrain policies, or
promote a new benchmark row.

## Evidence Sources

- Planner family matrix: `docs/benchmark_planner_family_coverage.md`
- Benchmark observation vocabulary: `robot_sf/benchmark/observation_levels.py`
- Planner observation/action metadata: `robot_sf/benchmark/algorithm_metadata.py`
- Fail-closed planner contract gate: `robot_sf/benchmark/planner_command_contract.py`
- Benchmark runner planner construction: `robot_sf/benchmark/map_runner.py`
- Default sensor-fusion observation contract: `docs/dev/observation_contract.md`
- LiDAR range sensor implementation: `robot_sf/sensor/range_sensor.py`
- Observation-level regression tests: `tests/benchmark/test_observation_levels.py`

## Current Contract Boundary

`lidar_2d` is already present as benchmark metadata. It means a range-sensor projection with
required inputs `robot_state`, `goal`, and `lidar_rays`. The current compatible observation modes
are `sensor_fusion_state` and `lidar_human_state`.

That metadata is not, by itself, proof that a planner is fair for a LiDAR-only benchmark. A fair
LiDAR track must also verify that the planner-facing observation is produced from rays and allowed
ego state, not from perfect `socnav_state`, simulator-backed occupancy grids, full map geometry, or
unoccluded pedestrian state.

The existing gate correctly fails some unsafe combinations. For example,
`tests/benchmark/test_observation_levels.py` expects `goal` with `observation_level="lidar_2d"` to
fail today because `goal` declares `goal_state` / `socnav_state`, not `sensor_fusion_state`.

## Planner Classification

| Planner family | Current input contract | LiDAR-track classification | Reason and caveat |
| --- | --- | --- | --- |
| Generic `ppo` metadata path | `sensor_fusion_state`: `robot_state`, `goal`, `lidar_rays`, `history` | Native metadata-compatible, not benchmark-proven | The metadata can resolve `lidar_2d`, but promoted PPO configs such as `configs/baselines/ppo_15m_grid_socnav.yaml` and many training configs use dict/grid SocNav observations. A LiDAR-only claim needs checkpoint provenance and an observation-key audit, or retraining. |
| `guarded_ppo` | `sensor_fusion_state` plus safety guard | Native metadata-compatible, not benchmark-proven | Policy input can be LiDAR-style, but the guard/fallback path must be audited so it does not consume privileged structured state. |
| `crowdnav_height` | `lidar_human_state`: `robot_state`, `goal`, `lidar_rays`, `humans` | Native metadata-compatible, experimental | The wrapper declares a LiDAR-plus-human contract, but it remains checkpoint/provenance-sensitive and experimental, not a first paper-facing baseline without source-harness proof. |
| `goal` | `goal_state` or ignored-extra `socnav_state` | Adaptable from LiDAR track metadata | The planner can run from robot/goal state and ignore rays, but current contracts reject `goal` plus `lidar_2d`. If included, label it as a goal-only control, not obstacle-aware LiDAR navigation. |
| `social_force`, `orca`, `hrvo`, SocNav ORCA variants | Structured robot, goal, pedestrian pose/velocity/radius | Adaptable from LiDAR-derived tracked agents | These planners are not raw-ray consumers. A fair LiDAR entry needs a ray-to-tracked-agent adapter with explicit noise/occlusion semantics. |
| `risk_dwa`, `mppi_social`, `hybrid_rule_local_planner`, `hybrid_portfolio`, `stream_gap` | Structured robot/goal/pedestrian state plus planner-specific local risk, TTC, lattice, or veto terms | Adaptable from LiDAR-derived tracked agents | These are plausible classical LiDAR-track candidates after a tracker adapter, but current structured observations would be privileged in a LiDAR-only run. |
| `grid_route`, `teb`, `safety_barrier` | Robot/goal plus local occupancy or corridor state | Adaptable from LiDAR-derived occupancy | The useful adapter is ray projection into an ego-frame local occupancy grid. It must not use simulator-backed global map grids or around-corner occupancy. |
| `prediction_planner`, `predictive_mppi`, `gap_prediction` | Structured state, history, and prediction-model features | Adaptable only after tracker history; partly training/provenance-bound | These should be second wave. The tracker history and predictive checkpoint provenance must be explicit before benchmark inclusion. |
| Promoted PPO grid/SocNav checkpoints and current SAC baselines | Dict `socnav_struct` / occupancy-grid observations in current configs | Training-required for LiDAR-only | Existing promoted configs are not enough for a fair LiDAR-only claim. A new LiDAR observation checkpoint and quality gate are required. |
| DreamerV3 drive-state/rays configs | RLlib `drive_state + rays` training profiles, no frozen planner row | Training-required / benchmark-row-required | Relevant training configs exist, but there is no promoted planner row or benchmark proof. |
| `socnav_sampling`, `sacadrl`, `socnav_bench`, Social-Navigation-PyEnvs wrappers | Legacy or external structured state and adapter-specific source contracts | Blocked or privileged without source-harness proof | Keep fail-closed unless dependency, checkpoint, and observation reconstruction proof exists. |
| `sonic_crowdnav`, GenSafeNav wrappers | GST/human-state checkpoint contracts | Blocked for LiDAR-only without source-specific proof | Human-state checkpoint contracts are not ray-only evidence. |
| `sicnav`, `dr_mpc`, `drl_vo` | External MPC or hybrid learned wrappers with dependency-sensitive state/action contracts | Blocked for first LiDAR track | These are useful research anchors but too adapter-heavy for the first fair LiDAR slice. |
| Placeholder `rvo` / `dwa` | Placeholder adapters | Excluded | Not benchmark-validated planner support. |

## Selected Adapter Follow-Ups

The first implementation wave should stay narrow:

1. Implement a `lidar_rays -> ego occupancy_grid` adapter for one occupancy-backed classical
   local-planner path, then smoke it with `grid_route`, `teb`, or `safety_barrier` behind
   testing-only guards.
2. Implement a `lidar_rays -> tracked_agents` adapter contract for social-state planners, then
   smoke one non-learning representative such as `orca`, `social_force`, or `risk_dwa`.

Issue #1615 covers learned LiDAR policy planning and feature extraction. Issue #1613 covers the
broader LiDAR benchmark setup and validation contract. The two adapter follow-ups above are the
concrete implementation slices selected by this audit.

Follow-up issues:

- https://github.com/ll7/robot_sf_ll7/issues/1659 - LiDAR-derived ego occupancy adapter.
- https://github.com/ll7/robot_sf_ll7/issues/1660 - LiDAR-derived tracked-agent adapter.

## Validation Policy

Before any planner is counted as LiDAR-track evidence:

- `validate_planner_contract(..., observation_level="lidar_2d")` must pass for the planner and
  fail for unsupported planner/level combinations.
- A smoke run must prove the planner-facing observation contains only allowed LiDAR-track inputs.
- Adapter metadata must state whether execution mode is native, adapter, degraded, fallback, or
  unavailable.
- Fallback and degraded execution must remain a caveat or exclusion, not benchmark-strengthening
  evidence.
- Testing-only planners still require explicit opt-in such as `allow_testing_algorithms: true`.

## Local Validation

Validation run on 2026-05-29:

```bash
PYTHONPATH=$PWD UV_PROJECT_ENVIRONMENT=/home/luttkule/git/robot_sf_ll7/.venv UV_NO_SYNC=1 \
  uv run pytest tests/benchmark/test_observation_levels.py \
    tests/benchmark/test_algorithm_metadata_contract.py -q
# 31 passed in 23.19s

PYTHONPATH=$PWD UV_PROJECT_ENVIRONMENT=/home/luttkule/git/robot_sf_ll7/.venv UV_NO_SYNC=1 \
  uv run python -c "from robot_sf.benchmark.planner_command_contract import validate_planner_contract; [validate_planner_contract(algo=a, robot_kinematics='differential_drive', algo_config={}, observation_level='lidar_2d') for a in ('ppo', 'guarded_ppo', 'crowdnav_height')]"
# lidar metadata candidates ok
```

The first command checks the fail-closed observation-level contract. The second command checks only
metadata compatibility for the native-candidate planners; it does not prove benchmark quality.
