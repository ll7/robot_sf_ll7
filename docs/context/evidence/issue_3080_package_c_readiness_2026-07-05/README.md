# Issue #3080: Prediction Package C Readiness Preflight

**Claim boundary:** coordination preflight only; no benchmark campaign executed and no forecast or paper-facing performance claim made

- **Overall status:** `ready`
- **Seed plan (same-seed):** [111, 2868]
- **Output roots:** ['docs/context/evidence/issue_2915_forecast_baselines_2026-06-20', 'docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13', 'docs/context/evidence/issue_2916_forecast_risk_coupling_2026-06-23']
- **Coupling result store available:** False
- **Coupling report available:** True
- **Coupling report path:** docs/context/evidence/issue_2916_forecast_risk_coupling_2026-06-23/forecast_risk_coupling_gate_report.json

## Arms

| arm | variant | risk_source | baseline | status |
| --- | --- | --- | --- | --- |
| no_forecast | none | none | - | `ready` |
| cv | cv | constant_velocity | constant_velocity_gaussian_baseline | `ready` |
| semantic_cv | semantic | semantic_cv | semantic_cv_baseline | `ready` |
| interaction_aware | interaction_aware | interaction_aware_cv | interaction_aware_cv_baseline | `ready` |
