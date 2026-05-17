# Issue #1290 Lazy Pygame Initialization

## Scope

Issue #1290 removes pygame/SDL import and initialization from the headless
`make_robot_env(debug=False)` path. Rendering behavior remains available through
`env.sim_ui` when debug or video recording is requested, but the actual
`SimulationView` is now materialized only when a visual attribute or `render()`
is used.

## Implementation Notes

- `robot_sf.render.sim_state` now owns the pygame-free `VisualizableAction` and
  `VisualizableSimState` dataclasses.
- `robot_sf.render.lazy_sim_view.LazySimulationView` preserves the existing
  non-`None` `sim_ui` handle for debug/video configurations while deferring
  `robot_sf.render.sim_view` import and `pygame.init()`.
- Optional visual helpers in `occupancy_grid`, `image_sensor`,
  `telemetry.visualization`, and bundled `pysocialforce` now import pygame only
  when their visual code paths are used.

## Validation Evidence

- Red first:
  `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/test_lazy_pygame_init.py -q`
  initially failed because pygame was imported during both `debug=False` and
  `debug=True` env creation.
- Lazy-init regression tests:
  `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/test_lazy_pygame_init.py -q`
  passed with `2 passed`.
- Adjacent rendering/recording tests:
  `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/test_pygame_headless.py tests/test_image_sensor.py tests/test_image_sensor_fusion.py tests/test_jsonl_recording.py tests/factories/test_recording_integration.py tests/factories/test_normalization.py tests/factories/test_incompatible_combinations.py -q`
  passed with `44 passed`.
- Import probe:
  after `from robot_sf.gym_env.environment_factory import make_robot_env` and
  `make_robot_env(debug=False)`, `"pygame" in sys.modules` stayed `False`.
- Cold/warm smoke, same command before and after:
  `LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python -m robot_sf.benchmark.perf_cold_warm --cold-runs 1 --warm-runs 1 --episode-steps 3 ...`
  recorded cold `env_create_sec` moving from `5.7666s` to `5.4446s` on this
  machine. The remaining cold cost is dominated by map/backend/JIT startup, so
  this smoke is supportive rather than a stable microbenchmark.

## Remaining Limitations

- Image observations, occupancy-grid pygame rendering, telemetry surface
  conversion, and explicit `env.render()` still import pygame by design.
- `perf_cold_warm` starts timing after module import, so the deterministic proof
  for the issue contract is the isolated import probe rather than the noisy cold
  `env_create_sec` sample alone.
