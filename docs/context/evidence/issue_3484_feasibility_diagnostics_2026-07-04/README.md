# Issue #3484 Feasibility Diagnostics Evidence Packet

This packet is diagnostic-only evidence for universally failing classic scenario families:
`bottleneck`, `cross_trap`, and `head_on_corridor`. It uses the issue #3484 feasibility
diagnostic runner and feeds lane outcomes into the existing `scenario_failure_cause.v1`
classifier. It is not benchmark-ranking evidence.

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

- Source tree before committing packet refresh: `origin/main` at `4cc7cbd66`.
- Report schema: `scenario_feasibility_diagnostics.v1`.
- Classifier schema consumed in `family_verdicts`: `scenario_failure_cause.v1`.
- Claim boundary: `diagnostic_only_not_benchmark_evidence`.
- Scenario source: `configs/scenarios/classic_interactions.yaml`.
- Scenario rows: 9.
- Generated locally on CPU; no Slurm, GPU, or campaign submission was used.
- Claim-boundary synthesis: `claim_boundary_synthesis.json`, generated from this retained packet
  without rerunning scenarios.

## Family Verdict Summary

| Family | Route feasible | Actor-free solved | Extended-time solved | Oracle/scripted solved | Diagnostic verdict | Ranking comparable |
| --- | --- | --- | --- | --- | --- | --- |
| `bottleneck` | `true` | `false` | `false` | `false` | `vehicle_infeasible` | `false` |
| `cross_trap` | `true` | `false` | `false` | `false` | `vehicle_infeasible` | `false` |
| `head_on_corridor` | `true` | `true` | `false` | `false` | `dynamic_blocking_or_deadlock` | `false` |

## Difficulty-Ramp Summary

The refreshed packet includes a diagnostic-only `difficulty_ramp` summary built from observed
scenario variants already run in the packet. It does not run a new benchmark campaign.

| Family | First actor-free failure | First oracle/scripted failure |
| --- | --- | --- |
| `bottleneck` | `medium` | `medium` |
| `cross_trap` | `low` | `low` |
| `head_on_corridor` | none observed | `medium` |

## Claim-Boundary Synthesis

`claim_boundary_synthesis.json` is the compact interpretation surface over the retained packet and
difficulty-ramp rows. It is fail-closed: missing verdicts, incomplete classifier inputs, unsupported
source schemas, or missing ramp rows become `still_unsupported`.

| Family | Claim state | Ranking comparable | Ramp boundary |
| --- | --- | --- | --- |
| `bottleneck` | `vehicle_infeasible` | `false` | actor-free and scripted/oracle first fail at `medium` |
| `cross_trap` | `vehicle_infeasible` | `false` | actor-free and scripted/oracle first fail at `low` |
| `head_on_corridor` | `dynamic_blocked` | `false` | actor-free has no observed failure; scripted/oracle first fails at `medium` |

No target family remains unsupported in this retained packet, but the synthesis remains
diagnostic-only and not ranking evidence.

## Boundaries

These verdicts are diagnostic proxy evidence for scenario-family closure triage. They do not change
benchmark rankings, planner comparisons, leaderboard rows, manuscript claims, or dissertation
claims. This run does not include a full benchmark campaign, Slurm/GPU submission, ranking update,
or paper-facing interpretation.
