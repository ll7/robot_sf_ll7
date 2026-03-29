# Issue 549: Station platform benchmark map rationale

## What this adds

`maps/svg_maps/classic_station_platform.svg` adds a station/platform-inspired classic interaction
scenario with a visible train/track edge, a long narrow platform, repeated door-side columns, and
concourse-side furniture/kiosk pinch points. The robot keeps to the concourse edge while the main
pedestrian stream uses the platform band closer to the train.

## Why it belongs in the benchmark set

- It expands the classic interaction matrix with a real-world-inspired elongated passing corridor.
- It exercises asymmetric obstacle placement rather than the symmetric bottleneck and crossing
  layouts already in the pack.
- It uses the existing single-pedestrian `wait_at` contract for a small deterministic waiting
  component, which is plausible for a station platform without inventing a new crowd model.
- It keeps the benchmark contract simple: one canonical layout, parser-safe labels, and direct
  inclusion in `configs/scenarios/classic_interactions.yaml`.

## Behavioral boundary

This issue does not implement general route-population waiting or stochastic dwell-time modeling.
What is included is the already-supported deterministic waiting path for selected
`single_pedestrians`, which is enough to capture a limited "pause near a train door" behavior.
Broader station-crowd behavior would be a separate modeling issue.

## Validation path

- Structural parser proof:
  `uv run pytest -x tests/maps/test_station_platform_map.py tests/test_svg_classic_maps_format.py`
- Scenario-matrix proof:
  `uv run pytest -x tests/test_classic_interactions_matrix.py`
- Smoke path:
  `uv run python scripts/validation/performance_smoke_test.py --scenario configs/scenarios/archetypes/classic_station_platform.yaml --num-resets 1`
