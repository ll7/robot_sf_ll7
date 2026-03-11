# Issue 495 Execution Notes

## Scope

Implement a lightweight overall performance trend layer on top of the existing cold/warm checks.

## Implemented

- Added matrix-based trend runner:
  - `robot_sf/benchmark/perf_trend.py`
- Added stable matrix definition:
  - `configs/benchmarks/perf_trend_matrix_classic_v1.yaml`
- Updated nightly workflow to run the matrix and compare with recent history snapshots:
  - `.github/workflows/perf-nightly.yml`
- Added unit tests for matrix loading, scenario execution wiring, history comparison,
  and `main()` flow:
  - `tests/benchmark/test_perf_trend.py`
- Extended docs with local + CI usage and interpretation guidance:
  - `docs/performance_notes.md`

## Validation Commands

```bash
uv run ruff check robot_sf/benchmark/perf_trend.py tests/benchmark/test_perf_trend.py
uv run pytest -x tests/benchmark/test_perf_trend.py tests/benchmark/test_perf_cold_warm.py
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

## Notes

- Trend history comparison is optional and driven by `--history-glob`.
- Nightly workflow restores cached history reports when available, then stores the current run
  report back into history for future comparisons.
