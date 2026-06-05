# Issue #2307 Topology Score Diagnostic

Status: current, diagnostic-only.

Issue #2307 reran the topology primary-route diagnostic after the Issue #2282 score-selection
instrumentation. The goal was to explain whether `primary_route` dominance came from missing
alternatives, invalid alternatives, score over-selection, or topology irrelevance.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2
```

The diagnostic completed on commit `d9ce46a9d1ee8197bbb8116a02378ee1e5f55308`. Raw trace output
remains ignored locally; compact evidence is tracked in
`docs/context/evidence/issue_2307_topology_score_diagnostic_2026-06-05/summary.json`.

## Result

Classification: `scoring_overselects_primary`.

| Field | Value |
| --- | --- |
| Steps | 160 |
| `ok` hypothesis frames | 90 |
| Insufficient-hypothesis frames | 70 |
| Steps with scored alternatives | 98 |
| Route-selector selected hypotheses | `primary_route`: 97, `masked_cell_87_79`: 1 |
| Topology-command influence hypotheses | `primary_route`: 33 |
| Rejection reason | `lower_topology_selection_score`: 98 |
| Alternative score-margin median | 0.4056854309944198 |
| Alternative score-margin max | 8.214213684774414 |
| Hypothesis switch count | 0 |

The single non-primary route-selector choice happened at step 35, where `masked_cell_87_79` beat
`primary_route` by about `1.8e-15`, effectively a numerical tie. The local command source on that
step was `dynamic_window`, and `topology_command_influence` stayed null, so it was not a corrective
topology switch.

## Interpretation

The strongest explanation is score-surface over-selection of the primary route, not absence of
alternatives. Alternatives were generated and scored often enough for proposal availability to be
an incomplete explanation, and rejection reasons consistently identify lower topology selection
score.

This does not prove a topology mitigation or benchmark improvement. It narrows the next useful
work: revise topology scoring, add a route-diversity/commitment term, or construct a falsification
case where a non-primary route should clearly win before tuning downstream command selection.
