# Issue #1322 SocNavBench Device-Placement Cleanup

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1322>

## Decision

The `tfe.DEVICE_PLACEMENT_SILENT` references in vendored SocNavBench were stale TensorFlow-era
comments around code that is now NumPy-native. There were no active `.numpy()` conversions or
TensorFlow device-placement calls to remove in the inspected paths.

The safe cleanup is therefore intentionally narrow:

* remove the stale device-placement comments from angle-wrapping paths,
* simplify the misleading voxel-map index variable that claimed an `int64` cast while already using
  NumPy `int32` indexing,
* add focused NumPy behavior tests rather than attempting a TensorFlow rewrite.

## Boundary

This does not make SocNavBench TensorFlow-native. A broader TF conversion would need a separate
runtime parity contract for gradients, tracing, dtype/device behavior, and upstream compatibility.

Direct `ruff check` on explicit `third_party/socnavbench/...` files still exposes the existing
vendored lint/docstring backlog. Repository-level Ruff keeps `third_party` excluded, so validation
uses focused behavior tests plus the normal PR-readiness gate.

## Validation

Targeted checks run on 2026-05-18:

```bash
uv run pytest tests/test_socnavbench_numops.py tests/test_planner/test_socnav.py -k socnavbench -q
uv run ruff check tests/test_socnavbench_numops.py
uv run ruff format --check tests/test_socnavbench_numops.py
rg -n "DEVICE_PLACEMENT_SILENT" third_party/socnavbench || true
```

The focused test run produced `6 passed, 1 skipped, 12 deselected`. The skip is the existing
upstream-data-dependent SocNavBench end-to-end test.
