# Issue #1415 Capability-Aware Map Resolver

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1415>

## Goal

Issue #1415 adds fail-closed map capability checks to scenario `map_id` resolution while preserving
legacy v1 registry behavior and direct `map_file` parser tests.

## Implemented Contract

`robot_sf/training/scenario_loader.py` now loads v2 catalog rows as resolved entries with:

- local SVG path,
- declared capabilities,
- source hash,
- role/profile,
- limitations,
- validation status.

When a scenario references `map_id`, the loader defaults to requiring `robot_runtime`. Scenario or
benchmark manifests can request another supported profile with `required_map_profile` or
`map_profile`, including `benchmark_candidate`, `route_only`, `pedestrian_runtime`, or
`obstacle_source`.

Failure messages name the requested profile, missing capability, map path, validation status,
role/profile, and limitations. If a v2 row has a stale `source_sha256`, resolution fails before the
scenario is accepted.

## Boundaries

Generated converted-map cache loading remains out of scope. Legacy v1/simple registries do not
provide capabilities and continue to resolve as path lookups for compatibility.

## Validation

Targeted proof for this slice:

```bash
uv run ruff check robot_sf/training/scenario_loader.py tests/training/test_scenario_loader.py tests/training/test_scenario_loader_additional.py
uv run pytest tests/training/test_scenario_loader.py tests/training/test_scenario_loader_additional.py -q
uv run pytest tests/training/test_scenario_loader.py tests/training/test_scenario_loader_additional.py tests/maps -q
uv run python scripts/validation/check_docs_proof_consistency.py --base origin/issue-1413-map-catalog-schema-sync
```
