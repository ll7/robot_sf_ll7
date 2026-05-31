# Candidate Report: hybrid_rule_v3_progress_2p4_static_escape_probe (smoke)

## Decision

pass

## Hypothesis

Keeping the 2.4 m/s progress envelope while enabling the static-escape and recenter probes from the faster static-escape candidate may convert some low-progress stalls without adopting the 3.0 m/s fast-progress speed envelope.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_progress_2p4_static_escape_probe_smoke_pr1837_current_head/summary.json`
- Git commit: `2a14055d5621522f6663b20d72784bd37b3b14e5`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.8700 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior, generalization, or planner superiority.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |

## Family Override Runs

- `nominal`: success `1.0000`, collision `0.0000`
