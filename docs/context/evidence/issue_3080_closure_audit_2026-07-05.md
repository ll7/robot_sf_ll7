# Issue #3080 Closure Audit

Issue #3080 asked for Package C coordination across open-loop prediction baselines, observation replay, and closed-loop forecast-risk coupling: compare no-forecast, constant-velocity (CV), semantic-CV, and interaction-aware variants under identical seeds, keep average displacement error and final displacement error (ADE/FDE) secondary, and stop before learned-predictor expansion unless the simple baselines leave a preregistered closed-loop gap. This audit maps the acceptance criteria to merged PR evidence and finds the issue closable at the diagnostic Package C coordination boundary.

## Closure Decision

Status: `complete`

Closure keyword for the handoff PR: `Closes #3080`

Claim boundary: diagnostic Package C coordination only. This audit does not promote predictor accuracy, planner rankings, benchmark Results claims, paper claims, dissertation claims, or a full campaign result. The tracked artifacts are single-fixture diagnostic evidence and readiness evidence.

## Evidence Chain

| Step | Evidence | Status |
| --- | --- | --- |
| Open-loop forecast analysis | Issue #2915 is closed. The tracked evidence bundle `docs/context/evidence/issue_2915_forecast_baselines_2026-06-20/` contains forecast batches for CV, semantic, and interaction-aware baselines plus comparison reports. | Met before closed-loop assembly. |
| Observation perturbation replay | Issue #2777 is closed. The tracked evidence bundle `docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13/` records diagnostic observation-noise replay evidence for the same occluded-emergence fixture. | Met as diagnostic replay input. |
| Closed-loop forecast-risk coupling | Issue #2916 is closed. PR #4551 regenerated `docs/context/evidence/issue_2916_forecast_risk_coupling_2026-06-23/forecast_risk_coupling_gate_report.json` with fixture `seed=111`, scenario `issue_2756_occluded_emergence`, four coupling rows, diagnostic claim boundary, and verdict `continue`. | Met as diagnostic coupling evidence. |
| Package C admission check | PRs #3744, #3875, #4542, and #4551 added and tightened `scripts/tools/prediction_package_c_readiness.py`, then committed `docs/context/evidence/issue_3080_package_c_readiness_2026-07-05/package_c_readiness_report.json`. The report has `overall_status: ready`, seed plan `[111, 2868]`, all four arms `ready`, `coupling_report_available: true`, and no coupling-report blockers. | Met. |

## Acceptance Criteria Mapping

| Acceptance criterion | Evidence | Closure assessment |
| --- | --- | --- |
| Uses ADE/FDE only as secondary metrics; primary outcomes include risk calibration, false stops, first yield, minimum separation, progress, near misses, collisions, and runtime. | The #2916 coupling report is explicitly about closed-loop outcome effects, not forecast accuracy. Its row metrics include `collision`, `near_miss`, `safety_events`, `min_distance_m`, `progress_m`, `first_yield_step`, `false_positive_stops`, `runtime_s`, and `snqi`; the verdict checks safety-event reduction without false-positive-stop or runtime regression. The #3080 readiness validator requires primary outcome fields including collision, near miss, progress, false-positive stops, runtime, and SNQI before admitting the report. | Met for diagnostic Package C coordination. ADE/FDE remain in the open-loop #2915 evidence, not the closure decision's primary outcome. |
| Runs open-loop forecast analysis before closed-loop coupling. | The readiness report lists the open-loop evidence root `docs/context/evidence/issue_2915_forecast_baselines_2026-06-20` before the closed-loop #2916 evidence root. Issue #2915 closed before the #2916 and #3080 readiness rerun artifacts used here. | Met. |
| Runs live observation perturbation replay forecast-risk coupling under same seeds. | The #3080 readiness report records seed plan `[111, 2868]` and validates the supplied #2916 report fixture seed `111` against the Package C seed plan. It also carries the observation replay evidence root `docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13`, and all four Package C arms are `ready`. | Met at the diagnostic replay/coupling boundary. |
| Stops before learned-predictor expansion unless simple baselines leave a preregistered closed-loop gap. | The #2916 report uses no learned predictor, is marked `diagnostic_only`, and states `paper_grade: false`. The rows cover only no-forecast, CV, semantic-CV, and interaction-aware CV variants. The verdict is `continue`, but the caveats keep the result single-fixture and non-promotional. | Met. No learned-predictor expansion is part of the closure evidence. |

## Validation Evidence

| Check | Result |
| --- | --- |
| `uv run python scripts/tools/prediction_package_c_readiness.py --coupling-report docs/context/evidence/issue_2916_forecast_risk_coupling_2026-06-23/forecast_risk_coupling_gate_report.json --output-json /tmp/issue-3080-readiness-verify.json` | Passed. The generated report has `overall_status: ready`, all four arms `ready`, `coupling_report_available: true`, and `coupling_report_blockers: []`. |
| `python3` JSON inspection of the #2916 report | Passed. The report has `issue: 2916`, `claim_boundary: diagnostic_only`, `paper_grade: false`, fixture seed `111`, scenario `issue_2756_occluded_emergence`, four rows, and verdict `continue`. |

## Residual Risks

| Risk | Why it remains |
| --- | --- |
| The issue labels still show stale blocked/proposal state at audit time. | GitHub label cleanup is state propagation, not repository evidence. The PR carrying this audit should close #3080 on merge. |
| The evidence is diagnostic, not paper-grade or campaign-scale. | The accepted Package C closure boundary is coordination/readiness plus single-fixture diagnostic coupling evidence. Full benchmark campaign runs, Slurm/GPU submissions, and paper/dissertation claim edits were explicitly out of scope for this ready-queue task. |
| The interaction-aware row degrades to plain CV mean in the single observed-pedestrian fixture. | The #2916 report records this caveat honestly, so it is not promoted as interaction-model evidence. |

No full benchmark campaign was run for this audit, no Slurm or GPU submission was made, and no paper/dissertation claim text was edited.
