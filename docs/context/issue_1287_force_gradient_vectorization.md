# Issue #1287 Force-Gradient Interpolation Vectorization

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1287>

## Goal

`force_gradient_norm_mean` samples a force-field gradient along the robot trajectory. Before this
change, it called scalar `_bilinear` once per robot position, making long benchmark episodes pay a
Python-loop cost inside metric aggregation.

## Decision

Keep the existing scalar `_bilinear` helper as the reference contract and add `_bilinear_many` as a
NumPy vectorized companion. `force_gradient_norm_mean` now computes all trajectory samples in one
call and filters finite samples before averaging.

The vectorized helper preserves the scalar behavior for:

- lower and upper grid boundaries,
- out-of-bounds samples,
- non-finite trajectory coordinates,
- degenerate grid spacing.

## Validation

Focused parity and metric tests:

```bash
./.venv/bin/python -m pytest tests/test_metrics.py -k "force_gradient or bilinear_many" -q
```

Result:

```text
3 passed, 63 deselected
```

Local interpolation microbenchmark using 200,000 in-bounds trajectory samples on a 160 x 120 grid:

```text
scalar_seconds=2.128318
vectorized_seconds=0.046518
speedup=45.75x
```

The benchmark also asserted `np.testing.assert_allclose(..., equal_nan=True, rtol=1e-12, atol=1e-12)`
between scalar-loop samples and vectorized samples.

## Follow-Up Boundary

This issue only removes the Python loop from force-gradient interpolation. Broader metric-module
profiling, import-time cleanup, or alternate interpolation backends should stay in separate issues.
