# Simulator-Error and Diagnostic Success Contract (Issue #8693)

**Status:** current / diagnostic-only implementation-integrity contract — it does not establish a
benchmark, planner-quality, safety, or failure-rate claim.
**Issue:** [#8693](https://github.com/ll7/robot_sf_ll7/issues/8693), following the four-way receipt
taxonomy in [#8686](https://github.com/ll7/robot_sf_ll7/pull/8686).
**Canonical implementation:** `robot_sf/benchmark/force_coupled_comparator.py`.
**Contract tests:** `tests/benchmark/test_force_coupled_comparator.py`.
**Receipt schema:** `robot_sf/benchmark/schemas/force_coupled_comparator_receipt.v1.json`.

Plain-language summary: the analytic comparator now records simulator-boundary failures as
`simulator` failures, keeps planner-owned lifecycle and output failures as `path_generation`
failures, and preserves the established path-generation fallback for recognized non-success rows
without a more specific cause.

## Canonical rollout contract

`execute_rollout` sets the simulator-error flag only when an exception occurs in analytic simulator
state preparation, clearance checks, or kinematic state integration. The resulting row is
structured as `status="error"`, `degraded=true`, with a `simulator_<phase>_failure` reason and
`failure_class="simulator"`. Planner reset, `planner.plan()`, diagnostics, and invalid command or
diagnostic output remain `path_generation` failures through the existing `plan_exception` reason
mechanism.

Explicit boundary flags identify ownership before reason-substring classification: an explicit
simulator flag yields `simulator`, while a planner exception yields `path_generation` even when
the exception text contains words such as `simulator`. For rows without an explicit boundary flag,
the remaining reason precedence is:

1. `simulator`
2. `social_compliance`
3. `tracking`
4. `path_generation`

Planner-owned diagnostic reasons carry a `planner_diagnostic` prefix, so reason text cannot
override that ownership even when it contains a simulator keyword. Unknown rollout statuses and
non-canonical classes fail closed with `ValueError`. For a recognized non-success status without a
more specific signal, `classify_failure` preserves the established `path_generation` fallback.
Summary aggregation rejects a missing non-success degradation reason or class, so a row cannot
silently disappear from `failure_class_counts`.

The rollout boundary deliberately catches broad `Exception` values so failures from the external
planner and analytic simulator phases become structured diagnostic rows instead of escaping the
comparator. These seven fault-isolation handlers are explicitly recorded in the repository's
broad-exception baseline; the baseline approval does not broaden the benchmark claim.

## Diagnostic success and provenance

This change does not redefine the existing `success_rate` metric: it remains based on completed,
collision-free rows. Near misses, degraded rows, and fallback reasons remain visible in their
separate status/diagnostic fields, but this issue does not promote a new strict-success or
failure-rate interpretation. Summary validation does require every non-`ok` row to retain its
emitted `degraded=true` invariant, at least one degradation reason, and a canonical failure class.

The canonical registry in this module contains native analytic planners only. This change does
not convert adapter, fallback, or degraded provenance into native evidence, and it makes no
benchmark-improvement claim. Any future adapter or fallback registration must retain explicit
execution metadata and pass the same fail-closed summary contract before its rows can be
interpreted.

The receipt remains `force_coupled_comparator_receipt.v1`; the taxonomy fields stay additive and
optional for legacy receipt compatibility. A legacy non-success row may omit `failure_class`, but
an explicitly present value must be a canonical non-null class; generated receipts carry the
explicit class and summary counts, and runtime aggregation remains fail-closed for missing
classes.
