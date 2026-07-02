# Issue #3484 feasibility diagnostics evidence

This directory contains a diagnostic-only smoke report for the universally failing classic scenario
families `bottleneck`, `cross_trap`, and `head_on_corridor`.

Command:

```bash
uv run python scripts/tools/run_feasibility_diagnostics_issue_3484.py \
  --scenario-config configs/scenarios/classic_interactions.yaml \
  --family bottleneck \
  --family cross_trap \
  --family head_on_corridor \
  --output docs/context/evidence/issue_3484_feasibility_diagnostics_2026-07-02/diagnostic_report.json
```

Summary from `diagnostic_report.json`:

| family | route feasible | actor-free solved | oracle/scripted solved | extended-time solved | provisional verdict |
| --- | --- | --- | --- | --- | --- |
| `bottleneck` | `true` | `false` | `false` | `null` | `vehicle_infeasible` |
| `cross_trap` | `true` | `false` | `false` | `null` | `vehicle_infeasible` |
| `head_on_corridor` | `true` | `true` | `false` | `null` | `indeterminate` |

Claim boundary: `diagnostic_only_not_benchmark_evidence`. This is not a full benchmark campaign,
not a planner ranking update, and not a paper-facing claim.
