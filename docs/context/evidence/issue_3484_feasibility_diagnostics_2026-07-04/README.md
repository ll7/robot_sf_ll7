# Issue #3484 Feasibility Diagnostics Evidence Packet

This packet is a diagnostic-only run for the universally failing classic scenario families:
`bottleneck`, `cross_trap`, and `head_on_corridor`. It uses the existing issue #3484 feasibility
diagnostic runner and feeds the resulting lane outcomes into the existing
`scenario_failure_cause.v1` classifier. It is not benchmark-ranking evidence.

## Command

```bash
uv run python scripts/tools/run_feasibility_diagnostics_issue_3484.py \
  --scenario-config configs/scenarios/classic_interactions.yaml \
  --family bottleneck \
  --family cross_trap \
  --family head_on_corridor \
  --include-extended-time \
  --output docs/context/evidence/issue_3484_feasibility_diagnostics_2026-07-04/diagnostic_report.json
```

## Provenance

- Source tree before committing this packet: `origin/main` at `d90c9cda7`.
- Report schema: `scenario_feasibility_diagnostics.v1`.
- Classifier schema consumed in `family_verdicts`: `scenario_failure_cause.v1`.
- Claim boundary: `diagnostic_only_not_benchmark_evidence`.
- Scenario source: `configs/scenarios/classic_interactions.yaml`.
- Scenario rows: 9.
- Generated locally on CPU; no Slurm, GPU, or campaign submission was used.

## Family Verdict Summary

| Family | Route feasible | Actor-free solved | Extended-time solved | Oracle/scripted solved | Diagnostic verdict | Ranking comparable |
| --- | --- | --- | --- | --- | --- | --- |
| `bottleneck` | `true` | `false` | `false` | `false` | `vehicle_infeasible` | `false` |
| `cross_trap` | `true` | `false` | `false` | `false` | `vehicle_infeasible` | `false` |
| `head_on_corridor` | `true` | `true` | `false` | `false` | `dynamic_blocking_or_deadlock` | `false` |

## Boundaries

The verdicts are diagnostic proxy evidence for scenario-family closure triage. They do not change
benchmark rankings, planner comparisons, leaderboard rows, manuscript claims, or dissertation
claims. The run does not include a difficulty-ramp expansion or a full benchmark campaign.
