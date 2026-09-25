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
  - `partial-failure`: some jobs failed or only part of the requested campaign completed; the run
    is not benchmark-success.
  - `failed`: the planner run failed.
  - `not_available`: the benchmark contract was not met, including fallback-only and skip cases.

## Benchmark Entry Point Policy

Runtime marker parsing follows declared field types. The shield's
`fallback_controller_state` is a diagnostic dictionary, not a fallback counter;
its presence alone does not establish fallback execution. It is admitted only
when the independently declared algorithm and all metadata identity fields agree
on guarded PPO. Direct parser calls without that binding reject the dictionary.
Availability uses the producer/config-bound `algorithm_readiness.name`; release
acceptance uses the manifest's expected arm. Missing or malformed identity cannot
grant the exception. Ordinary legacy summaries without shield state are unchanged.
The parser still scans its nested dictionaries and lists for forbidden statuses, true fallback/degraded
flags, and positive or malformed fallback counters. A non-dictionary value for
this field is invalid. Other fallback-named counter fields retain strict numeric
validation. Declared `mixed` command mode remains separate from fallback status.
Guarded PPO's safe Risk-DWA command is a declared component of that composite
planner. Release acceptance permits its exact `fallback_safe` counters only for
verified guarded identity; its exact `fallback_safe` decision label remains
admitted under that same identity-bound exception. The exact `stop_best_effort`
decision label is forbidden, and its counters admit only zero; positive or
malformed values fail closed. Other best-effort and uncertainty fallback
counters remain forbidden. This narrow label rule does not redefine safe shield
interventions as degraded execution.
Batch summaries retain the producer's `guard_stats` and
`shield_stats.decision_counts` across episodes. Valid identity-bound
`fallback_safe` counters are summed numerically; malformed counters remain
fail-closed. Any `stop_best_effort` counter or label is sticky across the
aggregate. Per-episode typed `last_decision` state is not merged; an aggregate
retains only the minimal stop label when needed to preserve failure evidence.

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
