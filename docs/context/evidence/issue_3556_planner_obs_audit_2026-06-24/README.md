# Benchmark planner observation-format audit (silent-blind-planner check)

**Question:** the #3556 work found that `stream_gap` was *silently blind* in the real benchmark
runner (it read the nested SOCNAV observation while `map_runner` feeds a flat observation, so it
extracted `robot=[0,0], n_peds=0` and drove blind while still emitting collision "results"). Does any
planner in the **headline 7-planner comparison** share this failure?

**Answer: no — the headline comparison is clean.**

## Method

The bug class is an observation-*format* mismatch: a planner reading nested keys (`obs["robot"]`,
`obs["pedestrians"]`) on the flat benchmark observation (`robot_position`, `pedestrians_positions`,
`goal_current`). Audited by tracing each headline planner's observation extractor.

## Finding

The headline planners (`prediction_planner`, `goal`, `social_force`, `orca`, `ppo`,
`socnav_sampling`, `sacadrl`) are built on the shared `SamplingPlannerAdapter` /
`OccupancyAwarePlannerMixin` base, whose extractor `robot_sf/planner/socnav_occupancy.py::_socnav_fields`
**explicitly handles both the nested and the flat observation** (its docstring: "Normalize SocNav
observation (nested or flattened)"):

```python
if "robot" in observation:           # nested SOCNAV
    robot_state = observation["robot"]; ...
else:                                 # flat benchmark observation
    pos = observation.get("robot_position", [0.0, 0.0]); ...
    ped_state = {"positions": observation.get("pedestrians_positions"), ...}
```

So all the classical headline planners see the robot pose and the real pedestrians in `map_runner`.

`stream_gap` was the **lone exception**: a standalone adapter that rolled its own *nested-only*
`_extract_state` and never used `_socnav_fields`. It is **not in the headline 7-planner comparison**,
so the headline table was never affected. `stream_gap`'s extractor was fixed to accept the flat
observation in the #3556 PR (#3567).

## Conclusion for the dissertation

The headline planner-ranking comparison is **not corrupted** by the silent-blind-planner failure
mode. The one planner that exhibited it (`stream_gap`) is outside the comparison and is now fixed.

## Regression guard (this PR)

`tests/benchmark/test_planner_observation_contract.py` pins the invariant going forward: the shared
classical-planner extractor must return a non-degenerate view of the flat benchmark observation (sees
the robot pose and the actual pedestrian count, not the `[0,0]`/empty blind default), and must stay
backward-compatible with the nested observation. `stream_gap`'s extractor has its own flat-observation
regression test in `tests/benchmark/test_scenario_belief_policy_hook_issue_3556.py`.

A behavioural "reacts to pedestrians" check was rejected as the guard: it false-positives on the
`goal` trivial reference planner, which ignores pedestrians by design. The contract checks *what the
planner sees*, not how it reacts.
