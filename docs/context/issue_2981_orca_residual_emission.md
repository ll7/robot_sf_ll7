# Issue #2981: ORCA residual mechanism-trace emission path

## Scope

Issue #2981 wires the diagnostic-only `orca_residuals` mechanism-trace producer into a reusable
script path:

```bash
uv run python scripts/tools/emit_orca_residual_mechanism_trace.py \
  --planner-decision-trace docs/context/evidence/issue_2981_orca_residual_emission_2026-06-17/orca_residuals_planner_decision_trace.v1.json \
  --output docs/context/evidence/issue_2981_orca_residual_emission_2026-06-17/orca_residual_mechanism_trace.jsonl \
  --format jsonl \
  --trace-uri docs/context/evidence/issue_2981_orca_residual_emission_2026-06-17/orca_residuals_planner_decision_trace.v1.json \
  --report docs/context/evidence/issue_2981_orca_residual_emission_2026-06-17/orca_residual_emission_report.json
```

The script consumes a tracked ORCA residual planner-decision trace fixture and emits
`mechanism_trace.v1` rows without relying on direct test-only calls.

## Evidence Boundary

- Classification: `diagnostic_only`.
- The generated rows prove that the scripted emission path can transform durable fixture input into
  schema-valid `mechanism_trace.v1` ORCA residual rows.
- This does not add benchmark comparator, scenario-matrix, success-rate, safety, or paper-facing
  outcome evidence.

## Tracked Evidence

- `docs/context/evidence/issue_2981_orca_residual_emission_2026-06-17/orca_residuals_planner_decision_trace.v1.json`
- `docs/context/evidence/issue_2981_orca_residual_emission_2026-06-17/orca_residual_mechanism_trace.jsonl`
- `docs/context/evidence/issue_2981_orca_residual_emission_2026-06-17/orca_residual_emission_report.json`

Focused validation:

- `uv run pytest tests/benchmark/test_mechanism_trace.py`
- `uv run pytest tests/tools/test_emit_orca_residual_mechanism_trace.py`
- `uv run ruff check robot_sf/benchmark/mechanism_trace.py tests/benchmark/test_mechanism_trace.py scripts/tools/emit_orca_residual_mechanism_trace.py tests/tools/test_emit_orca_residual_mechanism_trace.py`
