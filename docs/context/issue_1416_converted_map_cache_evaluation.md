# Issue #1416 Converted-Map Cache Evaluation

## Decision

Do not add generated converted-`MapDefinition` disk cache support yet.

The existing in-process `_load_map_definition` LRU cache already covers the representative scenario
sets measured here without evictions. A new disk cache would add stale-artifact and invalidation
surface while the stacked catalog schema and capability-aware resolver PRs are still draft work.

## Issue Scope

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1416>

Related contract surfaces:

- `docs/context/issue_1348_capability_map_catalog_design.md`
- `docs/context/issue_1413_map_catalog_schema_sync.md`
- `docs/context/issue_1415_capability_map_resolver.md`
- `robot_sf/benchmark/perf_scenario_cache.py`
- `robot_sf/training/scenario_loader.py`

The issue asks for a measured decision first, with implementation only if justified. Because this
branch deliberately does not introduce a generated disk-cache runtime path, there is no new stale
cache or source-hash mismatch behavior to exercise. The cache hit, stale miss, and source-hash
mismatch test requirements remain the required proof for a future branch that actually implements
generated converted-map cache support.

## Measurements

Commands run from the `issue-1416-converted-map-cache-eval` worktree:

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python -m robot_sf.benchmark.perf_scenario_cache \
    --scenario-config configs/scenarios/classic_interactions.yaml \
    --repetitions 3 \
    --output-json output/benchmarks/perf/issue1416_classic_interactions_cache_profile.json \
    --output-markdown output/benchmarks/perf/issue1416_classic_interactions_cache_profile.md
```

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python -m robot_sf.benchmark.perf_scenario_cache \
    --scenario-config configs/scenarios/sets/ppo_all_available_training_v1.yaml \
    --repetitions 2 \
    --output-json output/benchmarks/perf/issue1416_ppo_all_available_training_cache_profile.json \
    --output-markdown output/benchmarks/perf/issue1416_ppo_all_available_training_cache_profile.md
```

Results:

| Scenario config | Scenarios | Unique maps | Cache maxsize | Hits | Misses | Evictions | Hit rate | Warm avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `configs/scenarios/classic_interactions.yaml` | 23 | 13 | 256 | 56 | 13 | 0 | 81.16% | 6.33 ms |
| `configs/scenarios/sets/ppo_all_available_training_v1.yaml` | 90 | 44 | 256 | 136 | 44 | 0 | 75.56% | 6.59 ms |

Compact tracked evidence: `docs/context/evidence/issue_1416_converted_map_cache_2026-05-20/summary.json`.

## Rationale

The measured scenario sets fit comfortably inside the current 256-entry LRU cache. The larger PPO
training set covers 44 unique maps, so a generated disk cache would not currently reduce repeated
SVG conversion inside a long-running process beyond the existing warm-cache behavior.

The cold misses are expected: one miss per unique map. The warm timings are already small relative
to training and benchmark work, and the profiler reports zero evictions in both representative
sets. That makes disk-cache complexity hard to justify today.

Keeping the source SVG plus parser/catalog versions canonical remains the safer boundary. If a
future campaign demonstrates cold-start map parsing as a bottleneck, the cache should still use the
policy from `docs/context/issue_1348_capability_map_catalog_design.md`: key by SVG bytes, parser
version, catalog schema version, and normalized parser options; include source path, source hash,
parser version, catalog schema version, generated timestamp, and requested capability/profile; and
treat stale or missing entries as misses rather than benchmark evidence.

Since no disk cache is added here, stale or missing generated cache entries cannot alter runtime or
benchmark behavior. The existing source-SVG path remains canonical and fail-closed capability
enforcement stays in the resolver layer.

## Artifact Decision

The raw profiler outputs are ignored `output/` artifacts and are not durable dependencies. They were
summarized into a compact tracked evidence bundle because the raw outputs are cheap to reproduce and
not needed by downstream runtime code.

## Validation

Validation commands:

```bash
uv run pytest tests/benchmark/test_perf_scenario_cache.py tests/maps -q
```

```bash
uv run python scripts/validation/check_docs_proof_consistency.py --base origin/issue-1415-capability-map-resolver
```

PR readiness should still run before handoff:

```bash
PYTEST_NUM_WORKERS=8 BASE_REF=origin/issue-1415-capability-map-resolver scripts/dev/pr_ready_check.sh
```
