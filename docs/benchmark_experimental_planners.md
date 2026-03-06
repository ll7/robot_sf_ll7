# Experimental Planner Guardrails

This repository now distinguishes between:

- `baseline-ready`: allowed in baseline-safe and paper-facing benchmark profiles
- `experimental`: allowed in exploratory benchmark profiles
- `experimental-testing`: planners that are implemented and tested, but should not silently enter
  normal benchmark sweeps until they have demonstrated stable benchmark value

## Testing-only planners

The following planners are currently treated as testing-only and fail closed by default:

- `risk_dwa`
- `mppi_social`
- `predictive_mppi`
- `hybrid_portfolio`
- `stream_gap`
- `gap_prediction`

These planners are blocked unless their algo config explicitly contains:

```yaml
allow_testing_algorithms: true
```

This keeps unfinished planner families available for controlled R&D while preventing accidental
inclusion in camera-ready or routine exploratory benchmark runs.

## Why this guard exists

These planners have unit/integration coverage and can be exercised locally, but their benchmark
performance is still unstable or materially below the current champion policy family. The guard
prevents:

- accidental inclusion in broad benchmark matrices
- accidental regression interpretation from incomplete planners
- confusing benchmark tables where low-readiness planners appear alongside promotion candidates

## Promotion rule

A testing-only planner should not lose the guard until it has:

1. repeatable benchmark evidence on the intended suite
2. contradiction-free outputs
3. documented failure modes and runtime limits
4. a clear reason to exist alongside current baselines

When those conditions are met, update `robot_sf/benchmark/algorithm_readiness.py` and remove the
opt-in requirement for that planner.
