# Issue #1291 PedestrianEnv Consolidation

## Scope

Issue #1291 removes the transition split between
`robot_sf/gym_env/pedestrian_env.py` and
`robot_sf/gym_env/pedestrian_env_refactored.py`. The full implementation now
lives directly in `robot_sf/gym_env/pedestrian_env.py`; the `_refactored` file
is removed.

## Implementation Notes

- `RefactoredPedestrianEnv` was renamed to `PedestrianEnv`.
- `RefactoredPedestrianEnv` and `PedestrianEnvRefactored` remain aliases inside
  the canonical module so external imports from `robot_sf.gym_env.pedestrian_env`
  that used transition-era class names still resolve.
- Tests and docs no longer import or link to `pedestrian_env_refactored.py`.

## Validation Evidence

- Red first:
  `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/test_pedestrian_env_consolidation.py -q`
  failed because `PedestrianEnv.__module__` pointed at
  `robot_sf.gym_env.pedestrian_env_refactored` and the `_refactored` file existed.
- Focused pedestrian/factory proof:
  `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/test_pedestrian_env_consolidation.py tests/test_pedestrian_env_compat.py tests/test_env.py tests/test_environment_factory_signatures.py tests/factories -q`
  passed with `55 passed`.
- Type-check proof:
  `uvx ty check robot_sf/gym_env/pedestrian_env.py tests/test_pedestrian_env_compat.py tests/test_pedestrian_env_consolidation.py --exit-zero`
  passed with `All checks passed!`.
- Lint proof:
  `uv run ruff check robot_sf/gym_env/pedestrian_env.py tests/test_pedestrian_env_compat.py tests/test_pedestrian_env_consolidation.py docs/refactoring/README.md docs/refactoring/refactoring_summary.md docs/2x-speed-vissimstate-fix/README.md docs/extract-pedestrian-action-helper/README.md`
  passed.

## Remaining Notes

- The issue-specified `tests/test_pedestrian_env.py` path does not exist on the
  current `origin/main`; the validation used the existing pedestrian compatibility,
  environment, and factory tests instead.
