# Issue 549: Station platform benchmark map rationale

## What this adds

`maps/svg_maps/classic_station_platform.svg` adds a station/platform-inspired classic interaction
scenario with an asymmetric platform aisle: a blocked upper edge (train/platform boundary), a clear
robot corridor near the concourse edge, and bench/column obstacles that force pedestrians to weave
instead of simply traversing an empty rectangle.

## Why it belongs in the benchmark set

- It expands the classic interaction matrix with a real-world-inspired elongated passing corridor.
- It exercises asymmetric obstacle placement rather than the symmetric bottleneck and crossing
  layouts already in the pack.
- It keeps the benchmark contract simple: one canonical layout, parser-safe labels, and direct
  inclusion in `configs/scenarios/classic_interactions.yaml`.

## Validation path

- Structural parser proof:
  `uv run pytest -x tests/maps/test_station_platform_map.py tests/test_svg_classic_maps_format.py`
- Scenario-matrix proof:
  `uv run pytest -x tests/test_classic_interactions_matrix.py`
- Smoke path:
  `uv run python scripts/validation/performance_smoke_test.py --scenario configs/scenarios/archetypes/classic_station_platform.yaml --num-resets 1`
