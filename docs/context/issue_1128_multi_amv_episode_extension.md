# Issue 1128 multi-AMV episode extension

Date: 2026-05-12

## Scope implemented

This branch adds a namespaced, additive multi-AMV episode extension block and updates the
existing multi-AMV smoke runner to emit canonical benchmark-style outputs:

- JSON summary for the smoke run,
- schema-validated episode JSONL,
- aggregate JSON with inter-robot metrics,
- compact Markdown report.

Implemented files:

- `robot_sf/benchmark/multi_amv.py`
- `scripts/validation/run_multi_amv_smoke.py`
- `tests/benchmark/test_multi_amv.py`

## Record shape

`multi_amv_episode_extension(...)` returns:

```json
{
  "multi_amv": {
    "enabled": true,
    "settings": {
      "num_robots": 2,
      "near_miss_distance_m": 1.0,
      "collision_distance_m": 0.4,
      "deadlock_speed_mps": 0.05,
      "deadlock_window_steps": 10
    },
    "planner_status": "goal_controller_smoke",
    "planner_support": {}
  }
}
```

The block is intentionally namespaced so existing single-robot episode consumers can ignore it without schema migration.
`planner_status` is required so callers keep the planner mode explicit, and the optional
`planner_note` field is omitted when no note is provided.

The canonical episode record carries inter-robot values under the standard metrics root:

```json
{
  "metrics": {
    "inter_robot": {
      "robot_count": 2.0,
      "pair_count": 1.0,
      "min_inter_robot_distance_m": 0.47,
      "inter_robot_collision_events": 0.0,
      "inter_robot_near_miss_events": 1.0,
      "deadlock_steps": 0.0,
      "deadlock_detected": false
    }
  }
}
```

`robot_sf.benchmark.aggregate.flatten_metrics(...)` flattens `metrics.inter_robot` into aggregate
rows, so `compute_aggregates(...)` and report outputs include the inter-robot metrics without
changing single-robot episode records.

## Fail-closed behavior

The helper raises `ValueError` when called with fewer than two robots or empty inter-robot metrics.

## Remaining work

- Broader `robot_sf_bench run` multi-AMV matrix support remains deferred until planner semantics
  are defined.
- Non-trivial planner-family support remains deferred to #1168 and must fail closed until a real
  fleet adapter exists.
- The current command path is still the explicit first-slice smoke runner, not a paper-facing
  comparable planner benchmark.

## Canonical smoke command

```bash
uv run python scripts/validation/run_multi_amv_smoke.py \
  --scenario configs/scenarios/single/multi_amv_minimal_smoke.yaml \
  --out output/benchmarks/issue_1128/multi_amv_smoke.json \
  --episodes-out output/benchmarks/issue_1128/episodes.jsonl \
  --aggregates-out output/benchmarks/issue_1128/aggregates.json \
  --report-out output/benchmarks/issue_1128/report.md \
  --horizon 10
```

Generated files under `output/benchmarks/issue_1128/` are disposable validation artifacts unless a
future issue explicitly promotes them as durable evidence.

## Validation status

Validation has been run for this implementation slice:

- `uv run pytest tests/benchmark/test_multi_amv.py -q`
- `uv run pytest tests/benchmark/test_camera_ready_campaign.py tests/benchmark/test_planner_command_contract.py -q`
- `uv run python scripts/validation/run_multi_amv_smoke.py --scenario configs/scenarios/single/multi_amv_minimal_smoke.yaml --out output/benchmarks/issue_1128/multi_amv_smoke.json --episodes-out output/benchmarks/issue_1128/episodes.jsonl --aggregates-out output/benchmarks/issue_1128/aggregates.json --report-out output/benchmarks/issue_1128/report.md --horizon 10`

These checks cover the additive namespaced record shape, fail-closed single-robot and
empty-metrics paths, schema-validated JSONL output, aggregate inclusion, and the smoke command
path.
