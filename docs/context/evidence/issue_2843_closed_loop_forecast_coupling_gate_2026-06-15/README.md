# Issue #2843 Closed-Loop Forecast Coupling Gate

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2843>

## Boundary

Diagnostic-only closed-loop forecast coupling gate. Not paper-facing evidence. No
learned training involved. Deterministic: no expensive campaign run.

## Inputs

- Forecast comparison: `docs/context/evidence/issue_2781_interaction_aware_forecast_2026-06-15/comparison_report.json`
- Gate readme: `docs/context/evidence/issue_1897_predictive_coupling_gate_2026-05-31/README.md`

## Forecast Interaction Effect

| Metric | Value |
|---|---:|
| Matched rows | 3 |
| Evaluable rows | 15 |
| Mean ADE 1s delta vs CV (m) | 0.0246 |
| Mean NLL 1s delta vs CV | -0.6717 |

Interaction-aware heuristic improved Gaussian likelihood/calibration proxy but worsened
point accuracy on matched diagnostic rows; revise before closed-loop coupling claims.

## Closed-Loop Gate

| Metric | Value |
|---|---|
| Status | `failed` |
| Reason | `global_success_delta_below_gate` |
| Global success delta | 0.0000 |
| Hard success delta | 0.0000 |
| Global min-distance delta | 0.0108 |

## Recommendation

**REVISE**

Forecast interaction_aware worsened 1s ADE by 0.0246 m vs CV and improved 1s NLL by
0.6717. Closed-loop gate failed (reason=global_success_delta_below_gate,
global_success_delta=0.0). Recommendation: revise forecast coupling before closed-loop
claims.
