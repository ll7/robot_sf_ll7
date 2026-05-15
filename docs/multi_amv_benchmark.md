# Multi-AMV Benchmark First Slice

Issue: [#1092](https://github.com/ll7/robot_sf_ll7/issues/1092)

The first multi-AMV benchmark slice is intentionally minimal. It proves that a scenario can declare
more than one robot, run through the existing `MultiRobotEnv` path with a simple supported goal
controller, and emit explicit inter-robot safety/deadlock metrics.

## Scenario Surface

```yaml
multi_amv:
  num_robots: 2
  near_miss_distance_m: 1.0
  collision_distance_m: 0.4
  deadlock_speed_mps: 0.05
  deadlock_window_steps: 8
```

The tracked smoke scenario is
`configs/scenarios/single/multi_amv_minimal_smoke.yaml`.

## Metrics

`robot_sf.benchmark.multi_amv.inter_robot_metrics` emits:

* `robot_count`
* `pair_count`
* `min_inter_robot_distance_m`
* `inter_robot_collision_events` for contiguous collision encounters, not raw collision steps
* `inter_robot_near_miss_events` for contiguous near-miss encounters, not raw near-miss steps
* `deadlock_steps`
* `deadlock_detected` when the entire fleet stays below the deadlock speed threshold for the
  configured window

These metrics are pairwise over robot positions only. They do not replace the existing
single-robot benchmark metrics. The explicit smoke command can emit them under the canonical
episode metrics root as `metrics.inter_robot`, while broad `robot_sf_bench run` multi-AMV matrix
support remains deferred until planner semantics are defined.

## Smoke Command

```bash
rtk uv run python scripts/validation/run_multi_amv_smoke.py \
  --scenario configs/scenarios/single/multi_amv_minimal_smoke.yaml \
  --out output/benchmarks/issue_1092/multi_amv_smoke.json \
  --horizon 40
```

The smoke uses a simple goal controller for every robot. It is meant to validate scenario parsing,
`MultiRobotEnv` execution, and metric emission, not to demonstrate fleet-optimal coordination.

To emit benchmark-style artifacts for #1128, add the canonical output paths:

```bash
rtk uv run python scripts/validation/run_multi_amv_smoke.py \
  --scenario configs/scenarios/single/multi_amv_minimal_smoke.yaml \
  --out output/benchmarks/issue_1128/multi_amv_smoke.json \
  --episodes-out output/benchmarks/issue_1128/episodes.jsonl \
  --aggregates-out output/benchmarks/issue_1128/aggregates.json \
  --report-out output/benchmarks/issue_1128/report.md \
  --horizon 10
```

`episodes.jsonl` is schema-validated, the aggregate JSON flattens `metrics.inter_robot` into
reportable metric keys, and the Markdown report summarizes the same inter-robot metric block.

## Limits

This first slice does not add centralized fleet optimization, multi-planner benchmark matrices, or
paper-facing comparable planner reporting. The main benchmark runner remains single-robot until
follow-up work defines planner-family semantics for multi-AMV records. Non-trivial planner support
is tracked in [#1168](https://github.com/ll7/robot_sf_ll7/issues/1168).
