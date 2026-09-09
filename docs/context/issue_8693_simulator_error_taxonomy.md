# Simulator-Error and Diagnostic Success Contract (Issue #8693)

**Status:** current / diagnostic-only implementation-integrity contract — it does not establish a
benchmark, planner-quality, safety, or failure-rate claim.
**Issue:** [#8693](https://github.com/ll7/robot_sf_ll7/issues/8693), following the four-way receipt
taxonomy in [#8686](https://github.com/ll7/robot_sf_ll7/pull/8686).
**Canonical implementation:** `robot_sf/benchmark/force_coupled_comparator.py`.
**Contract tests:** `tests/benchmark/test_force_coupled_comparator.py`.
**Receipt schema:** `robot_sf/benchmark/schemas/force_coupled_comparator_receipt.v1.json`.

Plain-language summary: the analytic comparator now records simulator-boundary failures as
`simulator` failures, keeps planner exceptions as `path_generation` failures, and refuses to
invent or drop a taxonomy class when a non-success rollout has no recognized cause.

## Canonical rollout contract

`execute_rollout` sets the simulator-error flag when an exception occurs during planner reset,
observation/clearance preparation, diagnostics, command conversion, or kinematic state
integration. The resulting row is structured as `status="error"`, `degraded=true`, with a
`simulator_<phase>_failure` reason and `failure_class="simulator"`. Exceptions raised by
`planner.plan()` remain `path_generation` failures because they occur at the planner boundary,
not in the comparator's simulator step.

Classification precedence remains explicit and four-way:

1. `simulator`
2. `social_compliance`
3. `tracking`
4. `path_generation`

Unknown rollout statuses, unrecognized non-success conditions, missing non-success classes, and
non-canonical classes fail closed with `ValueError`. There is no implicit fallback to
`path_generation`, and summary aggregation cannot silently omit a failure from
`failure_class_counts`.

## Diagnostic success and provenance

For this comparator, `success_rate` counts only rows that are all of the following: `status="ok"`,
not degraded, completed, collision-free, and not a near miss. A near miss may remain an `ok` row
with no failure class because it is a separate clearance caveat rather than one of the four
failure mechanisms, but it is never counted as diagnostic success. Degraded and fallback
execution are likewise excluded from success; their status and degradation reasons remain
visible rather than being normalized into a clean result.

The canonical registry in this module contains native analytic planners only. This change does
not convert adapter, fallback, or degraded provenance into native evidence, and it makes no
benchmark-improvement claim. Any future adapter or fallback registration must retain explicit
execution metadata and pass the same fail-closed summary contract before its rows can be
interpreted.

The receipt remains `force_coupled_comparator_receipt.v1`; the taxonomy fields stay additive and
optional for legacy receipt compatibility. Generated receipts carry the explicit class and
summary counts.
