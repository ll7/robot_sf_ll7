# Forecast-Risk Closed-Loop Coupling Gate (issue #2916)

## Claim boundary

- evidence_tier: `stress`
- claim_boundary: `diagnostic_only`
- paper_grade: `False`

Diagnostic-only forecast-risk closed-loop coupling gate on a single deterministic occluded-emergence fixture. Not paper-facing benchmark evidence. Tests whether forecast-derived risk changes navigation outcomes, not whether any predictor is accurate. No learned predictor is promoted.

## Verdict

**Decision: `continue`**

At least one forecast-risk row reduced safety events vs the control without a false-positive-stop or runtime regression. Forecast-risk coupling shows a navigation-outcome benefit worth pursuing.

## Reproducibility

- Issue: #2916
- Generated (UTC): 2026-07-05T01:04:08.184361+00:00
- Command: `uv run python scripts/benchmark/run_forecast_risk_coupling_gate.py --config configs/research/forecast_risk_coupling_issue_2916.yaml --output-dir docs/context/evidence/issue_2916_forecast_risk_coupling_2026-06-23`
- Repo HEAD: `816976e44`
- Config: `configs/research/forecast_risk_coupling_issue_2916.yaml`
- Fixture: `tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_episode_0000.json`
- Seed: 111

## Per-row outcomes

| row | class | collision | near_miss | safety_events | progress_m | stop_step | FP_stops | runtime_s | SNQI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_forecast | ok | False | True | 1 | 0.82 | None | 0 | 0.000303 | 0.273333 |
| cv_risk | ok | False | False | 0 | 0.45 | 5 | 0 | 0.001496 | 0.65 |
| semantic_risk | ok | False | False | 0 | 0.45 | 5 | 0 | 0.001337 | 0.65 |
| interaction_risk | ok | False | False | 0 | 0.45 | 5 | 0 | 0.001342 | 0.65 |

## Safety-benefit vs false-positive-stopping trade-off

- **cv_risk**: safety_event_reduction=1, FP_stops=0, benefit=True, regression=False
- **semantic_risk**: safety_event_reduction=1, FP_stops=0, benefit=True, regression=False
- **interaction_risk**: safety_event_reduction=1, FP_stops=0, benefit=True, regression=False

## Caveats

- Single deterministic fixture (seed=111), single occluded-emergence scenario.
- Closed-loop proxy: robot replays the fixture trajectory under a gate hold; no live planner or simulator is re-executed.
- interaction_risk degrades to plain CV mean (single observed pedestrian, no neighbor context); it is reported honestly, not as interaction evidence.
- Ground-truth pedestrian distance is used only for OUTCOME scoring, never as a risk source. Risk sources use the deployable tracked-agents observation tier.
- Not statistically powered; diagnostic_only.
