# Issue 483 Execution Notes

## Scope
Implement continuous cold/warm performance regression checks for classic interactions.

## Implemented
- Added reusable benchmark/check module: `robot_sf/benchmark/perf_cold_warm.py`.
- Added committed baseline snapshot:
  `configs/benchmarks/perf_baseline_classic_cold_warm_v1.json`.
- Added PR smoke gate in CI (`.github/workflows/ci.yml`):
  - one classic crossing scenario,
  - 1 cold sample + 2 warm samples,
  - relative regression thresholds and baseline enforcement.
- Added nightly scheduled workflow (`.github/workflows/perf-nightly.yml`):
  - broader run budget (2 cold + 4 warm),
  - two scenario densities (`classic_crossing_low`, `classic_crossing_medium`),
  - artifact upload for trend inspection.
- Added unit tests for median aggregation, baseline loading, comparison logic,
  diagnostics classification, and markdown report generation.

## Rationale
- Cold metrics capture startup/JIT/initialization shifts.
- Warm metrics capture steady-state simulation throughput shifts.
- Relative thresholds + absolute delta floors reduce false positives from noise.

## Artifacts
- PR smoke output:
  - `output/benchmarks/perf/cold_warm_pr_smoke.json`
  - `output/benchmarks/perf/cold_warm_pr_smoke.md`
- Nightly outputs:
  - `output/benchmarks/perf/nightly/*.json`
  - `output/benchmarks/perf/nightly/*.md`
