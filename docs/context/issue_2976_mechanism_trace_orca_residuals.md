# Issue #2976: mechanism_trace.v1 ORCA residual producer integration

## Chosen mechanism family
- `orca_residuals` is selected as the next single mechanism family.
- Rationale: it is a compact, high-value local-navigation intervention with structured decision-adaptation metadata (`action_adaptation`) already present in guarded-ORCA/PPO planner decisions, so a row emitter can be added without cross-module planner rewiring.
- Scope constraint: this PR adds only the producer + fixture-backed contract test; broader benchmarking/integration wiring remains a follow-up.

## Evidence emitted
- Fixture: `tests/benchmark/fixtures/orca_residuals_planner_decision_trace.v1.json`
- Contract tests:
  - `test_emit_orca_residual_row`
  - `test_emit_orca_residual_rows_from_fixture`
- Executable validation command:
  - `uv run pytest tests/benchmark/test_mechanism_trace.py -q`

## Claim status
- `diagnostic_only`: emits schema-valid `mechanism_trace.v1` rows and row-classification contracts from durable test fixture data.
- No benchmark comparator or scenario-matrix claim is made in this change.
