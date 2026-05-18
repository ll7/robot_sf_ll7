# Issue #1286 SNQI Bootstrap Stability

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1286>

## Goal

Remove the benchmark-facing `bootstrap_stability()` placeholder so it cannot be mistaken for
evidence-grade SNQI stability output.

## Decision

SNQI bootstrap ranking stability is now supported only for episode records that already contain
finite `metrics.snqi` values. The helper does not recompute SNQI from raw components because this
API does not carry normalization baselines. Missing SNQI values, missing planner group keys,
single-group inputs, invalid sample counts, and missing RNG control fail closed with `ValueError`.

## Implementation

`robot_sf.benchmark.snqi.bootstrap.bootstrap_stability(...)` now:

* groups episodes by `group_key` (`algo` by default),
* computes the baseline mean-SNQI ranking,
* performs deterministic stratified bootstrap resampling with caller-provided RNG,
* computes Spearman correlation of each resampled ranking against the baseline ranking,
* returns `status: "ok"` with normalized stability in `[0, 1]`.

The deprecated `--ci-placeholder` metadata path in `scripts/snqi_weight_optimization.py` no longer
points at `bootstrap_stability()` as a future scaffold. It remains explicit placeholder metadata and
directs callers to computed `--bootstrap-samples` output.

## Validation

Focused validation:

```bash
./.venv/bin/python -m pytest tests/test_snqi/test_bootstrap_stability.py -q
./.venv/bin/python -m pytest tests/test_snqi/test_bootstrap_stability.py tests/test_snqi_fixture_minimal.py tests/test_snqi/test_bootstrap_intervals.py tests/test_snqi/test_bootstrap_cli.py -q
./.venv/bin/ruff check robot_sf/benchmark/snqi/bootstrap.py scripts/snqi_weight_optimization.py tests/test_snqi/test_bootstrap_stability.py
./.venv/bin/ruff format --check robot_sf/benchmark/snqi/bootstrap.py scripts/snqi_weight_optimization.py tests/test_snqi/test_bootstrap_stability.py
uvx ty check robot_sf/benchmark/snqi/bootstrap.py tests/test_snqi/test_bootstrap_stability.py --exit-zero
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```

The tests cover deterministic non-placeholder output, missing `metrics.snqi`, and single-group
failure behavior. Adjacent SNQI bootstrap CLI tests passed to confirm existing score-interval output
still works.
