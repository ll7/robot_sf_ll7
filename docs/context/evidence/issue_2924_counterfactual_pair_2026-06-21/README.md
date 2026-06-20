# Issue #2924 Counterfactual Pair Evidence

Analysis-only local slice for the counterfactual-pair runner.

## Claim Boundary

- Evidence tier: `analysis_only`.
- This bundle is not benchmark-strength, paper-grade, or planner-superiority evidence.
- Fallback, degraded, failed, partial-failure, and `not_available` rows fail closed in the runner.

## Reproduction

```bash
scripts/dev/run_worktree_shared_venv.sh -- python scripts/analysis/run_counterfactual_pair_issue_2924.py \
  --manifest configs/research/issue_2924_counterfactual_pair.yaml \
  --output docs/context/evidence/issue_2924_counterfactual_pair_2026-06-21/result.json \
  --markdown-output docs/context/evidence/issue_2924_counterfactual_pair_2026-06-21/report.md \
  --panel-output-dir docs/context/evidence/issue_2924_counterfactual_pair_2026-06-21/panels
```

## Inputs

- `configs/research/issue_2924_counterfactual_pair.yaml`: `counterfactual_pair.v1` manifest.
- `baseline_result.json` and `intervention_result.json`: compact observation fixtures.
- `baseline_mechanism_trace.json` and `intervention_mechanism_trace.json`: `mechanism_trace.v1` activation fixtures.
- `baseline_trace_export.json` and `intervention_trace_export.json`: compact
  `simulation_trace_export.v1` fixtures used for deterministic diagnostic panels.

## Result

The tracked pair is prediction-risk related. The intervention activates `prediction_risk_gating`,
the active-row delta is expected to be positive, and `min_clearance_m` is expected to increase.
The generated `result.json` and `report.md` record the survived/falsified decision after the
runner is executed.
