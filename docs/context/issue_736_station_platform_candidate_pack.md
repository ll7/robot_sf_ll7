# Issue 736: Station-platform candidate pack

## Decision

Use `configs/scenarios/sets/station_platform_candidate_pack_issue736.yaml` as an exploratory
pack on top of the issue #549 station-platform map. Do not add these variants to the default
`classic_interactions.yaml` matrix until a benchmark run shows non-redundant signal.

## Candidate variants

| Scenario | Purpose | Keep condition |
| --- | --- | --- |
| `station_platform_route_flow_low` | Low-density route-flow baseline on the platform geometry. | Keep only if it differs from corridor or bottleneck controls. |
| `station_platform_static_furniture_low` | Static passenger markers plus the existing obstacle/furniture layout. | Keep only if platform-edge clearance failures differ from doorway/bottleneck cases. |
| `station_platform_waiting_passengers_medium` | Deterministic dwell behavior near stair access using existing `wait_at` support. | Best candidate for distinct station-platform coverage. |
| `station_platform_dense_stress_optional` | High-density stress probe. | Optional only; reject if runtime or failure semantics are unstable. |

## Canonical commands

Structural contract:

```bash
uv run pytest tests/test_station_platform_candidate_pack.py tests/maps/test_station_platform_map.py -q
```

Smoke path:

```bash
uv run python scripts/validation/performance_smoke_test.py \
  --scenario configs/scenarios/sets/station_platform_candidate_pack_issue736.yaml \
  --num-resets 1 \
  --json-output output/benchmarks/issue736_station_platform_pack_smoke.json
```

Full exploratory benchmark path:

```bash
uv run python scripts/run_classic_interactions.py \
  --scenario-matrix configs/scenarios/sets/station_platform_candidate_pack_issue736.yaml \
  --output output/benchmarks/issue736_station_platform_pack/episodes.jsonl \
  --workers 1 \
  --horizon 120 \
  --no-resume
```

## Local pilot evidence

Executed on 2026-05-02:

```bash
uv run python scripts/run_classic_interactions.py \
  --scenario-matrix configs/scenarios/sets/station_platform_candidate_pack_issue736.yaml \
  --output output/benchmarks/issue736_station_platform_pack/episodes.jsonl \
  --workers 1 \
  --horizon 120 \
  --no-resume
```

Observed summary:

- 12/12 jobs completed in native `goal` baseline mode.
- `benchmark_success=true`; no fallback or degraded planner mode was used.
- 0/12 episodes reached success within the short 120-step pilot horizon.
- Static and dense variants produced early `collision` statuses on 5/12 episodes.

Implication: the pack is runnable, but the short pilot is diagnostic only. It supports retaining the
pack as an exploratory candidate, not promoting station-platform variants into the paper matrix.

## Interpretation boundary

The pack is a benchmark-shaping probe, not a station-navigation claim. Treat "not distinct enough"
as a valid outcome. Platform-specific metrics are not added here; candidate observations are limited
to existing success, collision, near-miss, comfort, timeout, and runtime metrics.
