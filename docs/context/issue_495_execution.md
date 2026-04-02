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

## Follow-up Context: 2026-04-02 Nightly Failure

Run:
- GitHub Actions `Nightly Performance` run `#48`
- URL: `https://github.com/ll7/robot_sf_ll7/actions/runs/23883543583`

Observed result:
- Workflow setup, dependency sync, cache restore, and artifact upload all succeeded.
- The only failing step was `Run nightly trend benchmark matrix` in
  `.github/workflows/perf-nightly.yml`.
- `robot_sf.benchmark.perf_trend` exited non-zero because the history comparison status was
  `fail`.

Artifact-backed findings:
- Scenario-level `perf_cold_warm` reports for all three matrix scenarios were only `warn`, not
  `fail`.
- The cold-path diagnostics were startup-only (`env_create_sec` / `first_step_sec`) and already
  matched the intended advisory semantics from the underlying cold/warm layer.
- The top-level `perf_trend` failure came from the historical comparison for
  `classic_crossing_low` warm metrics:
  - `warm.first_step_sec`
  - `warm.episode_sec`
  - `warm.steps_per_sec`
- The downloaded artifact at `nightly-perf-artifacts` showed a noisy low-scenario warm sample pair:
  - one warm sample near `1119 steps/sec`
  - one warm sample near `182 steps/sec`
  - with `warm_runs=2`, the median became `650.5 steps/sec`, which was enough to trip the
    historical regression gate against prior history near `1176 steps/sec`.

Interpretation recorded for future discussion:
- This looked more like nightly measurement instability than a clear repo-wide performance
  regression.
- There is an important policy mismatch between layers:
  - `robot_sf.benchmark.perf_cold_warm` already treats startup-only regressions as `WARN`
  - `robot_sf.benchmark.perf_trend` still turns any history comparison regression into a hard
    `fail`
- The immediate recommendation at the time was to discuss stabilizing nightly measurement first
  (for example, more warm samples or another outlier-resistance strategy) before weakening the
  historical regression gate globally.

Why keep this note:
- The failure is not urgent enough to force an immediate policy change.
- The reasoning is worth preserving so later discussion can distinguish:
  - real trend regressions,
  - startup-only advisory noise,
  - and nightly instability caused by too few warm samples.
