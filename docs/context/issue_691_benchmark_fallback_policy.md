# Issue #691 Benchmark Fallback Policy

This note defines the canonical benchmark-facing fallback policy for Robot SF.

## Rule

- Fallback execution is allowed for explicit diagnostics and exploratory probes.
- Fallback execution is **not** acceptable as a successful benchmark outcome.
- If a planner or dependency cannot satisfy the required runtime contract, benchmark mode must fail
  closed with an explicit non-success status and a clear reason.

## Canonical Status Semantics

- `execution_mode`
  - `native`: planner executes in its intended benchmark contract.
  - `adapter`: planner executes through a declared compatibility adapter.
  - `mixed`: planner combines native and adapted execution semantics.
  - `unknown`: runtime contract could not be resolved.
- `readiness_status`
  - `native`: benchmark-capable without caveat.
  - `adapter`: benchmark-capable through a declared adapter.
  - `fallback`: runtime only became available through fallback behavior.
  - `degraded`: runtime was skipped, failed, or otherwise could not satisfy the contract cleanly.
- `availability_status`
  - `available`: benchmark-success capable.
  - `partial-failure`: some jobs failed; the run is not benchmark-success.
  - `failed`: the planner run failed.
  - `not_available`: the benchmark contract was not met, including fallback-only and skip cases.

## Benchmark Entry Point Policy

- `robot_sf_bench run` must return non-zero for:
  - `fallback`
  - `degraded`
  - `partial-failure`
  - `failed`
  - `not_available`
- `scripts/tools/run_camera_ready_benchmark.py` must return non-zero whenever the campaign contains
  any planner row that is not benchmark-success.
- Camera-ready reports must surface `availability_status`, `benchmark_success`, and
  `availability_reason` so benchmark readers can see why a planner was excluded or caveated.

## Diagnostic Exception

Diagnostic fallback remains valid for:

- source-harness probes,
- dependency reproduction scripts,
- exploratory integration notes where the purpose is to understand the failure boundary.

Those artifacts must still label fallback behavior explicitly as non-benchmark evidence.
