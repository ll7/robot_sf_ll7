# Issue 3142 Fast-PySF Force Optimization

Issue: <https://github.com/ll7/robot_sf_ll7/issues/3142>

## Summary

This note records a bounded fast-pysf force-computation optimization trial for
Issue #3142. The change caches pedestrian count once per
`GroupCoherenceForceAlt.__call__` invocation, then caches positions, groups, and
the configured force factor only after the no-group early return. The NumPy force
math and simulation semantics are unchanged.

## Evidence

Same-contract command shape:

```bash
uv run python scripts/validation/performance_smoke_test.py \
  --large-crowd-profile \
  --step-profile \
  --num-resets 2 \
  --step-samples 20 \
  --step-profile-mode steady \
  --step-profile-limit 8
```

The compact durable summary is tracked at
`docs/context/evidence/issue_3142_fast_pysf_force_optimization_summary.json`.
The raw JSON/telemetry files were generated under ignored `output/issue-3142/`
and should be treated as local disposable artifacts.

## Result Boundary

Baseline steady loop: `0.011563539505004883 s`, `1643.095523803633` steady
steps/sec.

Post-change steady loop: `0.011503934860229492 s`, `1651.6087956726285` steady
steps/sec.

The direction is slightly favorable at the whole steady-loop level, but the
specific `GroupCoherenceForceAlt.__call__` hotspot is flat to slightly worse
within the same single-run profile (`0.003266096 s` to `0.003273789 s`
cumulative time). Treat this as diagnostic supporting evidence for a tiny
semantics-preserving cleanup, not as a robust benchmark speedup claim.

## Validation

- `uv run pytest fast-pysf/tests/test_forces.py fast-pysf/tests/test_simulator.py`
  passed.
- `uv run pytest fast-pysf/tests` passed.
- `uv run pytest tests/perf/test_large_crowd_step_profile_contract.py tests/validation/test_performance_smoke_test.py`
  passed.
- `uv run ruff check fast-pysf/pysocialforce/forces.py` passed.
