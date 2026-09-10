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
`failure_class="simulator"`. Exceptions from planner reset, `planner.plan()`, or
`planner.diagnostics()`, as well as invalid command or diagnostic payloads, use the existing
`plan_exception:` reason mechanism. A valid diagnostic response with `status="degraded"`,
`degraded=true`, or `fallback=true` is not an exception: it remains a degraded row and records its
reason with the `planner_diagnostic:` prefix. Combined collision or simulator signals from either
kind of planner-owned diagnostic still follow the base taxonomy precedence below.

The base taxonomy precedence is preserved for combined signals. An explicit simulator flag or
simulator reason text is checked first, followed by social compliance, tracking, and path
generation:

1. `simulator`
2. `social_compliance`
3. `tracking`
4. `path_generation`

Consequently, `planner_diagnostic: simulator backend unavailable` with a pedestrian collision
remains `simulator`, while `plan_exception` with a pedestrian collision remains
`social_compliance`, matching the base simulator-first and social-before-path ordering. Unknown
rollout statuses and non-canonical classes fail closed with `ValueError`. For a recognized
non-success status without a more specific signal, `classify_failure` preserves the established
`path_generation` fallback. Summary aggregation rejects a missing non-success degradation reason
or class, so a row cannot silently disappear from `failure_class_counts`.

The rollout boundary deliberately catches broad `Exception` values so failures from the external
planner and analytic simulator phases become structured diagnostic rows instead of escaping the
comparator. These seven fault-isolation handlers are explicitly recorded in the repository's
broad-exception baseline; the baseline approval does not broaden the benchmark claim.

## Diagnostic success and provenance

For this comparator, the established v1 `success_rate` predicate remains exactly
`completed and not collision`. It intentionally does not filter by `status`, `degraded`, or
`near_miss`; those caveats remain visible through their dedicated fields and rates. A stricter
clean-completion slice would need a separately named or versioned metric and is outside this
repair. Summary validation does require every non-`ok` row to retain its emitted `degraded=true`
invariant, at least one degradation reason, and a canonical failure class.

The canonical registry in this module contains native analytic planners only, and its registry key
is passed into `execute_rollout` before any simulator-boundary work. Thus simulator failures that
occur before `planner.diagnostics()` remain grouped under the canonical planner ID rather than the
planner class name. This change does not convert adapter, fallback, or degraded provenance into
native evidence, and it makes no benchmark-improvement claim. Any future adapter or fallback
registration must retain explicit execution metadata and pass the same fail-closed summary
contract before its rows can be interpreted.

The receipt remains `force_coupled_comparator_receipt.v1`; the taxonomy fields stay additive and
optional for legacy receipt compatibility. A legacy non-success row may omit `failure_class`, but
an explicitly present value must be a canonical non-null class; generated receipts carry the
explicit class and summary counts, and runtime aggregation remains fail-closed for missing
classes.
