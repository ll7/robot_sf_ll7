# Issue #2923 Mechanism Trace v1 Schema

**Status**: current schema-contract slice.
**Issue**: [#2923](https://github.com/ll7/robot_sf_ll7/issues/2923)
**Branch/PR**: `issue-2923-mechanism-trace-schema`

## Scope

Issue #2923 introduces `mechanism_trace.v1`, a small source contract for local-navigation
intervention rows. The schema is intended to let mechanism-oriented reports compare rows from
static recentering, topology guidance, prediction-risk gating, ORCA residuals, signal-state logic,
and AMV actuation constraints without treating local `output/` files as durable evidence.

The first implementation adds:

- `robot_sf/benchmark/schemas/mechanism_trace.schema.v1.json`
- `robot_sf/benchmark/mechanism_trace.py`
- `tests/benchmark/fixtures/mechanism_trace.v1.example.json`

## Contract

Each row records:

- `mechanism_id`
- `activation_step`
- `input_condition`
- `selected_command`
- `command_source`
- `risk_score`
- `route_progress_delta`
- `failure_mode`
- `trace_uri`
- `classification`

The v1 classification vocabulary is:

- `inactive`
- `active-but-irrelevant`
- `slice-local`
- `revise`
- `stop`

The checked-in example fixture is a tracked fixture and source-contract example. It is not a
benchmark claim, paper-facing result, or durable evidence copy. `trace_uri` values in fixture rows
use repository-relative fixture anchors so they do not imply that worktree-local `output/` files are
durable dependencies.

## Initial Emitters

The first slice includes row emitters for two existing mechanism families:

- static recentering planner diagnostics via `emit_static_recentering_row`
- topology guidance planner diagnostics via `emit_topology_guidance_row`

The remaining mechanism IDs are schema-supported but still need concrete producer integrations
before they can support mechanism-level reports.

## Validation

Initial validation path:

```bash
uv run pytest tests/benchmark/test_mechanism_trace.py -q
uv run ruff check robot_sf/benchmark/mechanism_trace.py tests/benchmark/test_mechanism_trace.py
uv run python scripts/validation/check_docs_proof_consistency.py
git diff --check
```

## Follow-Ups

- Wire additional producers for `prediction_risk_gating`, `orca_residuals`,
  `signal_state_logic`, and `amv_actuation_constraints`.
- Use `mechanism_trace.v1` rows in mechanism reports against durable, schema-checked input before
  using them for benchmark-facing or paper-facing claims.
