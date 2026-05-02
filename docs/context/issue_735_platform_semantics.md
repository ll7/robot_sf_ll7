# Issue 735: Platform Semantics Boundary

## Decision

Add the smallest viable platform semantic representation as scenario-side metadata:
`platform_semantics`.

This is intentionally not parser-native yet. The first supported representation validates platform
hazard and keep-clear regions in YAML, preserves them for provenance, and fails closed when a
scenario declares that planner or metric consumers are required.

## Supported Contract

`platform_semantics.status`:

- `metadata_only`: validate and preserve the regions; no planner or metric behavior is claimed.
- `require_consumers`: reject the scenario in `build_robot_config_from_scenario` until explicit
  planner/metric consumers exist.

Region kinds:

- `hazard`
- `keep_clear`

Region shapes:

- `polygon`, with at least three `[x, y]` points.
- `bbox`, with `[min_x, min_y, max_x, max_y]`.

## Unsupported Boundary

- SVG label parsing for platform semantics is not implemented.
- Planner cost-map behavior is not implemented.
- Metric penalties or station-specific safety metrics are not implemented.
- `metadata_only` regions must not be described as behaviorally enforced.

## Validation

```bash
uv run ruff check robot_sf/training/scenario_loader.py tests/training/test_scenario_loader.py
uv run pytest tests/training/test_scenario_loader.py -q
uv run pytest tests/test_ai_prompt_surfaces.py -q
git diff --check
```

Focused RED/GREEN path:

- Before implementation, `test_load_scenarios_rejects_invalid_platform_semantics` failed because
  invalid `kind: escalator` was silently accepted.
- After implementation, scenario-loader tests pass and `require_consumers` fails closed before
  benchmark config construction.
