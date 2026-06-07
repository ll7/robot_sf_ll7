# Issue #2438 Static-Recenter Activation Closure

Issue: [#2438](https://github.com/ll7/robot_sf_ll7/issues/2438)
Date: 2026-06-07
Status: diagnostic-only closure from existing durable evidence.

## Goal

Resolve the static-recenter activation evidence gap named by Issue #2438 without rerunning the
held-out smoke when the required fields already exist in tracked evidence.

Issue #2438 asked whether static recentering activated in the held-out cases and, if it did,
whether it changed command source, progress, trajectory, or terminal outcome. The relevant
instrumented rerun already exists in Issue #2306 and the requested field mapping already exists in
Issue #2402.

## Evidence Reused

Source diagnostic:
[issue_2306_static_recenter_activation_trace.md](issue_2306_static_recenter_activation_trace.md)

Field-mapped decision:
[issue_2402_static_recenter_activation_decision.md](issue_2402_static_recenter_activation_decision.md)

Compact Issue #2438 bundle:
[evidence/issue_2438_static_recenter_activation_closure_2026-06-07/README.md](evidence/issue_2438_static_recenter_activation_closure_2026-06-07/README.md)

No new simulator rerun was performed because the #2306 rerun used the same held-out smoke contract:

- Baseline candidate: `hybrid_rule_v3_fast_progress`
- Intervention candidate: `issue_2170_static_recenter_only`
- Scenario matrix: `configs/scenarios/sets/issue_2128_heldout_family_transfer_pilot_eval.yaml`
- Seed: `111`
- Horizon: `500`

## Activation Result

| Scenario | Activation count | First activation step | Selected command source | Command source changed | Progress delta after activation | Trajectory delta | Terminal outcome changed | Classification |
| --- | ---: | --- | --- | --- | --- | ---: | --- | --- |
| `classic_station_platform_medium` | `0` | `null` | `[]` | `false` | `null` | `0.0 m` | `false` | `mechanism_inactive` |
| `francis2023_intersection_wait` | `0` | `null` | `[]` | `false` | `null` | `0.0 m` | `false` | `comparator_already_solved_case` |

Overall Issue #2438 decision: `mechanism_inactive`.

The unsolved held-out row is the discriminating row. Static recentering did not activate there, so
the unchanged failure is not evidence for active-but-irrelevant recentering. The solved row is kept
as secondary context because the baseline already succeeded and therefore offered no rescue target.

## Continue / Revise / Stop

Recommendation: `stop_current_heldout_transfer_route`.

Do not tune or promote static recentering from the current held-out smoke. Continue only if a new
issue predeclares a slice where the static-recenter gate should activate, records the same
activation fields, and explains why that slice targets a static-obstacle recentering failure mode
rather than pedestrian-flow or already-solved behavior.

## Claim Boundary

This is diagnostic-only synthesis. It is not benchmark-strength evidence, not held-out transfer
evidence, not a planner-improvement claim, and not a paper-facing mechanism panel. It closes a
missing-field loop by pointing Issue #2438 at the existing activation rerun and preserving a compact
reviewable table under `docs/context/evidence/`.

## Validation

Cheap validation for this docs/evidence closure:

```bash
rtk python -m json.tool docs/context/evidence/issue_2438_static_recenter_activation_closure_2026-06-07/summary.json
rtk python scripts/validation/check_research_lane_states.py
rtk python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_2438_static_recenter_activation_closure.md \
  --path docs/context/evidence/issue_2438_static_recenter_activation_closure_2026-06-07/README.md \
  --path docs/context/evidence/issue_2438_static_recenter_activation_closure_2026-06-07/summary.json \
  --path docs/context/catalog.yaml
rtk git diff --check
```
