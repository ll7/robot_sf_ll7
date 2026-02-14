# Warning Hygiene Sweep (2026-02-13)

Related issue: https://github.com/ll7/robot_sf_ll7/issues/496

## Baseline
Previous full test run emitted 10 warning-summary entries:
- 7 advisory `UserWarning`s from `tests/test_classic_interactions_matrix.py`
- 1 `RuntimeWarning` (`invalid value encountered in divide`) from `fast-pysf/pysocialforce/scene.py`
- 1 `DeprecationWarning` from jsonschema metaschema fallback in research schema validation
- 1 scipy `RuntimeWarning` (precision loss) in paired t-test fixture

## Approach
1. Fix root causes first; avoid blanket global warning filters.
2. For intentional scenario outliers, require explicit annotation instead of runtime warnings.
3. Add regression tests where warning-prone math paths are corrected.
4. Keep unresolved environment/toolchain warnings tracked explicitly (do not hide silently).

## Implemented Changes
- `fast-pysf/pysocialforce/scene.py`
  - Reworked `capped_velocity` to use `np.divide(..., where=desired_speeds > 0.0)`.
  - Prevents divide-by-zero invalid operations.
- `fast-pysf/tests/test_simulator.py`
  - Added regression test for zero desired-speed capping path.
- `robot_sf/benchmark/schemas/report_metadata.schema.v1.json`
  - Updated `$schema` URI to canonical draft 2020-12 URI.
- `specs/270-imitation-report/contracts/report_metadata.schema.json`
  - Same `$schema` URI update to keep contract/source aligned.
- `tests/research/test_statistics.py`
  - Updated paired t-test fixture to non-constant pairwise deltas (avoids unstable edge warning).
- `configs/scenarios/archetypes/classic_bottleneck.yaml`
  - Added `metadata.density_advisory: zero_baseline_route_spawn` to intentional zero-density baselines.
- `configs/scenarios/archetypes/classic_group_crossing.yaml`
  - Added `metadata.density_advisory: high_density_stress` to intentional high-density case.
- `tests/test_classic_interactions_matrix.py`
  - Replaced runtime advisory warnings with explicit annotation assertions.
  - Intentional outliers stay visible via config metadata, while test output remains high signal.
- `pyproject.toml` + `uv.lock`
  - Replaced `stable-baselines3[extra]` with `stable-baselines3` to avoid pulling
    `opencv-python` (and its SDL dylibs) into the default environment.
  - This removes the cv2/pygame SDL duplicate-library collision source on macOS.
- `docs/training/issue_403_grid_training.md`
  - Updated dependency note to reflect that Stable-Baselines3 is now part of core dependencies.

## Validation
- Targeted checks:
  - `uv run pytest tests/test_collision_sanity.py tests/research/test_schemas.py::test_metadata_schema tests/research/test_statistics.py::test_paired_t_test_basic tests/test_classic_interactions_matrix.py -q`
  - `uv run python -m pytest fast-pysf/tests/test_simulator.py -q`
- Full suite:
  - `uv run pytest tests -q`
  - Result: `1808 passed, 10 skipped` with no warnings summary block.

## Remaining / Deferred
- No known pytest warning-summary entries remain from the #496 checklist after these changes.
- Keep monitoring for environment-specific SDL diagnostics when users install additional local packages manually.
