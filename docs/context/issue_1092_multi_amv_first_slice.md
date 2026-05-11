# Issue #1092 Multi-AMV First Slice

Related issue: [#1092](https://github.com/ll7/robot_sf_ll7/issues/1092)

## Decision

Implement the maintainer-selected minimal first slice: one execution path, one scenario surface, and
one inter-robot metric block. Broader fleet coordination and main-runner integration stay deferred.

## Implemented Scope

* `multi_amv` scenario block with `num_robots` and inter-robot metric thresholds.
* `robot_sf.benchmark.multi_amv` settings parser and metric helper.
* `configs/scenarios/single/multi_amv_minimal_smoke.yaml`.
* `scripts/validation/run_multi_amv_smoke.py`, which runs `MultiRobotEnv` with a simple goal
  controller and writes inter-robot metrics.
* `MultiRobotEnv` now respects `config.map_id` instead of always selecting `uni_campus_big`.

## Deferred Scope

* Main `robot_sf_bench run` multi-AMV episode schema.
* Planner-family adapters for multi-robot coordination.
* Centralized fleet optimization.
* Aggregate reporting across multi-AMV benchmark matrices.

Main benchmark integration is tracked in
[#1128](https://github.com/ll7/robot_sf_ll7/issues/1128).

## Validation Plan

Use:

* `rtk uv run pytest tests/benchmark/test_multi_amv.py -q`
* `rtk uv run python scripts/validation/run_multi_amv_smoke.py --scenario configs/scenarios/single/multi_amv_minimal_smoke.yaml --out output/benchmarks/issue_1092/multi_amv_smoke.json --horizon 40`
* Full PR readiness before handoff.
