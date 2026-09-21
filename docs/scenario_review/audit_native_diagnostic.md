# BA-05 native diagnostic adapter

`robot_sf.analysis_workbench.audit_native_diagnostic` is a bounded, diagnostic-only
adapter for one admitted canonical-runner episode. It is a native execution probe,
not benchmark evidence and not a scientific evaluator.

## Input contract

The launcher supplies a `native-diagnostic.v1` request containing:

- an #9417 `component-request.v1` and `experiment-recipe.v1` pair;
- an external receipt and pinned local source root;
- one admitted `native-diagnostic-source.v1` source bundle; and
- one `robot_goal` intervention plus a finite child timeout.

The source bundle stores the historical original record and the closed runner input
allowlist. The adapter currently admits only `simple_policy`, explicit start/goal,
`analysis_trace=all`, and stateless execution. It also requires a canonical
`environment_identity` derived from the effective scenario, an
`initial_state_sha256` over the explicit robot start/goal, and a
`simple_policy-config.v1:<digest>` config identity. Those identities are stored
in the source document and recipe, carried by the historical record provenance,
checked against the trace and planner metadata, and retained in the spawned
runner provenance. Checkpoint, model, stateful, or arbitrary planner
configuration is rejected before a child starts. The receipt, source bytes,
request/recipe identities, source commit, and config identity are rechecked
after treatment; a changed source is unavailable rather than usable.

Every canonical record also carries one explicit role: `historical_original`,
`new_diagnostic_control`, or `new_diagnostic_intervention`. The role, source
identifier, and (for new records) diagnostic request identifier are checked
before fidelity or activation. Any fallback, degraded, adapter-active, or
evidence-ineligible marker in planner/native diagnostics fails closed, even when
the outer route still says `native`. This includes canonical nested
`execution_mode`/status fields and malformed `evidence_eligible` values; only a
typed boolean `true` is eligible, and every declared execution mode must be
exactly `native`. The runner's neutral `force_diagnostics` fallback fields and
optional analysis-trace coverage `unavailable` status are retained as known
non-planner limitations and do not weaken the native gate.

## Execution and result boundary

The control and intervention each call
`robot_sf.benchmark.runner.run_episode` in a spawned child with a parent-owned
deadline and cancellation path. A result is `complete` only when:

1. the historical original has essential native telemetry and the historical
   record role/source binding;
2. the new control has the control record role, request/source binding, native
   planner status, and valid trace telemetry with no fallback marker;
3. the control trace/outcome/status and complete canonical metric mapping match
   the original;
4. the intervention has the intervention record role/request/source binding and
   the changed goal produces a changed trace and measured paired trajectory delta;
5. the final source-admission check still passes.

`original_identity`, `control_identity`, and `intervention_identity` are distinct.
`fidelity` and `activation` contain measured digests and deltas. `unavailable`,
`failed`, and `cancelled` results are fail-closed and never become activation
evidence. Optional trace-coverage limitations remain visible in the result.

## True native smoke

Run the focused smoke from the repository root:

```text
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest -q \
  tests/analysis_workbench/test_audit_native_diagnostic.py::test_native_smoke_measures_control_and_intervention
```

The test first creates the original with the real canonical `run_episode` route,
then admits its exact source bytes and runs control/intervention through spawned
canonical children. Its result asserts the native route, source and runner-input
digests, distinct identities, exact control fidelity, and a non-zero measured
trajectory activation. The probe is diagnostic-only even when complete.

## Shared service and MCP boundary

`AuditService` exposes the same adapter as `run_native_diagnostic`. The launcher
configures one or more `NativeDiagnosticBinding` values containing the admitted
source root/receipt, component request, and recipe; each binding is checked
against the authenticated session `SourceRef`, policy `allowed_roots`, and
`allowed_recipes` before it is selected. The operation caller supplies only an
intervention ID, a two-coordinate robot goal, and a deadline no greater than
the adapter ceiling. Source paths, receipt digests, recipe identities, runner
inputs, planner names, and checkpoints are not accepted from browser/model
payloads.

The operation uses the shared durable service authority and `diagnostic.native`
receipt, including the admitted source digest and immutable context/source
revision. It preflights source bytes, the selected episode, historical
telemetry, planner eligibility, and source/recipe identity before charging the
finite aggregate compute budget or starting a native child. A kill switch sets
the owned cancellation event observed by the native child. Inflight operation
receipts remain conservative after a crash; duplicate operation IDs recheck
the admitted source and never start a second child. The closed MCP tool is
`run_native_diagnostic` and is available through both the offline dispatcher
and stdio transport.

Even a `complete` service result retains `evidence_boundary=diagnostic_only`
and `scientific_claim_allowed=false`; failed, negative, inconclusive, stale,
cancelled, and unavailable results are not benchmark evidence.

## Materialization boundary

The canonical retained `analysis_trace` is carried in each native runner record,
but this adapter does not own rendering. At this branch base,
`robot_sf/analysis_workbench/audit_materialize.py` is not present, so a
cross-module `materialize_episode` smoke cannot be run without coupling another
lane's implementation. Integration must verify that the retained control or
intervention trace can render a missing video through the materializer and must
keep lazy materialization or divergent replay distinct from this native result.
