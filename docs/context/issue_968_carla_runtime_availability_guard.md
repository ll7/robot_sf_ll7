# Issue #968 CARLA Runtime Availability Guard

## Scope

Issue #968 extends the CARLA bridge availability surface with `require_carla()`, a strict guard for
future replay entry points that need the optional CARLA Python API. Normal imports of
`robot_sf_carla_bridge` remain CARLA-free; callers opt into the strict guard only when they are
about to use CARLA-dependent runtime code.

The guard raises `CarlaUnavailableError` with an actionable message when the `carla` module is not
importable. If `carla` is importable, the helper returns the imported module so replay code can
avoid duplicating optional import handling.

## Boundary

This change does not start a CARLA simulator, run replay, install CARLA, change dependency
metadata, or compare simulator metrics. It only establishes the fail-clear import boundary needed by
later replay entry points.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_runtime.py -q
```

Failed before implementation because `CarlaUnavailableError` and `require_carla` were absent.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_runtime.py -q
```

Passed after adding the strict guard: `2 passed`.
