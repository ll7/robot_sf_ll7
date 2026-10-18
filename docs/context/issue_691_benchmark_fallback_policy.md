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
Guarded PPO's shield is a declared component of that composite planner. For
verified guarded identity, its exact `fallback_safe`, `fallback_best_effort`,
`stop_safe` and `stop_best_effort` decision labels and counters in `guard_stats`
and `shield_stats.decision_counts` are method telemetry, recorded and reported
without triggering the forbidden-marker gate. Native counters must still be
nonnegative integers (booleans, floats and malformed values reject). Every other
planner, unbound identity or wrong expected arm retains the existing forbidden
marker rules; uncertainty/checkpoint fallback, nested failures and unknown
fallback counters remain forbidden.

Static summary enrichment uses the underlying `algorithm: ppo`, matching the
per-episode producer, while `canonical_algorithm` and `planner_contract.planner_id`
remain `guarded_ppo`. Runtime aggregation carries these identity fields and
preserves conflicting or incomplete shield identity as rejection evidence.
Batch summaries sum valid producer counters numerically and retain malformed
counters. Per-episode typed `last_decision` state is not merged; the aggregate
retains its existing sticky minimal `stop_best_effort` label for reporting. That
label is telemetry only under the same verified identity. These rules classify
method telemetry, not safety quality or retrospective release admission.

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
