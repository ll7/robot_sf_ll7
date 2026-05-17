# Issue #1304 Pedestrian Config Boundary

Issue: #1304
Base PR: #1302
Date: 2026-05-17

## Goal

Remove the remaining `PedestrianEnv` type escape that passed a legacy
`robot_sf.gym_env.env_config.PedEnvSettings` through `SingleAgentEnv` with a bare `Any` cast.

## Decision

`PedestrianEnv` keeps accepting legacy `PedEnvSettings` for backward compatibility, but normalizes
that object into `robot_sf.gym_env.unified_config.PedestrianSimulationConfig` before base
environment setup.

This keeps the public constructor compatible while making the canonical runtime boundary explicit:

- environment setup and `SingleAgentEnv` receive `PedestrianSimulationConfig`,
- legacy configs are copied into the unified dataclass through a small adapter,
- deprecated `peds_have_obstacle_forces` overrides are applied only after normalization, so the
  caller's unified config object is not mutated without deep-copying freshly-created defaults.

## Validation

Focused validation:

```bash
uv run pytest tests/test_pedestrian_env_compat.py -q
uv run pytest tests/test_env_util_additional_coverage.py tests/map_test.py -q
uv run ruff check robot_sf/gym_env/pedestrian_env.py tests/test_pedestrian_env_compat.py
uv run ruff format --check robot_sf/gym_env/pedestrian_env.py tests/test_pedestrian_env_compat.py
uvx ty check robot_sf/gym_env/pedestrian_env.py tests/test_pedestrian_env_compat.py --exit-zero
PYTEST_NUM_WORKERS=8 BASE_REF=origin/main DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  scripts/dev/pr_ready_check.sh
```

Observed focused proof:

- legacy config adaptation and no-`Any`-cast regression tests failed before the implementation,
- `tests/test_pedestrian_env_compat.py -q` passed after the adapter and review fix,
- adjacent env utility and map tests passed,
- Ruff and `ty` checks passed for the touched Python files.
- Full readiness passed after the review fix:
  `3634 passed, 10 skipped, 3 warnings in 438.78s`.
- Changed-file coverage stayed above the 80% minimum for the parent-plus-child diff:
  `robot_sf/gym_env/env_util.py` 99.0%, `robot_sf/gym_env/pedestrian_env.py` 86.4%,
  and `robot_sf/sim/simulator.py` 95.7%.

## Follow-Up Boundary

This issue does not retire legacy `PedEnvSettings` everywhere. The intentionally narrow boundary is
the `PedestrianEnv` constructor and base-environment setup path. Broader retirement of legacy config
classes should remain a separate migration issue with factory/docs call-site updates.
