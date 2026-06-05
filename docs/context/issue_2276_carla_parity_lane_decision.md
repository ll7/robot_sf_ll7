# Issue #2276 CARLA Parity Lane Decision

Issue: [#2276](https://github.com/ll7/robot_sf_ll7/issues/2276)
Parent issue: [#1491](https://github.com/ll7/robot_sf_ll7/issues/1491)
Related gated issues: [#1508](https://github.com/ll7/robot_sf_ll7/issues/1508),
[#1509](https://github.com/ll7/robot_sf_ll7/issues/1509),
[#1510](https://github.com/ll7/robot_sf_ll7/issues/1510),
[#1511](https://github.com/ll7/robot_sf_ll7/issues/1511)
Date: 2026-06-05
Status: analysis-only lane decision; not CARLA replay parity evidence.

Compact decision summary:
`docs/context/evidence/issue_2276_carla_parity_lane_decision_2026-06-05/summary.json`.

## Decision

Decision: `proceed_to_alignment_repair`.

The CARLA parity lane should not launch the #1510 multi-scenario bundle or draft the #1511
native/aligned parity claim yet. It should next repair or certify the native/aligned fixture
boundary: either make the certified scenario spawn without unplanned CARLA projection, or define
a pre-declared reversible aligned transform with an explicit threshold before replay.

## Evidence Reviewed

| Evidence | Observation | Decision impact |
| --- | --- | --- |
| [Issue #1444 coordinate alignment contract](issue_1444_carla_coordinate_alignment_contract.md) | Only `native` and threshold-bounded `aligned` replay modes are eligible for metric parity. `adapted`, `failed`, and `not-available` are diagnostic only. | Establishes the fail-closed gate. |
| [Issue #1508 eligibility audit](issue_1508_carla_native_aligned_eligibility.md) | No existing candidate is ready for a multi-scenario native/aligned campaign. The best metric-emitting row is smoke-only, not a certified scenario row. | Blocks #1510 until a durable fixture is certified. |
| [Issue #1509 fixture certification](issue_1509_carla_native_fixture_certification.md) | The `pr_promoted_planner_smoke` fixture is valid and CARLA Docker was available, but live replay mode was `oracle-replay-adapted`; coordinate replay mode was `adapted`. | Host availability is not the blocker; fixture alignment is. |
| `docs/context/evidence/issue_1509_carla_native_fixture_2026-05-31/live_replay_pr_promoted.json` | `coordinate_alignment.eligible_for_metric_parity` is `false`; robot spawn projection is `18.19085467444781 m`; replay status is `oracle-replay`. | Confirms the adapted replay is a fail-closed exclusion, not degraded success. |
| `docs/context/evidence/issue_1509_carla_native_fixture_2026-05-31/parity_report_pr_promoted.json` | Parity report status is `unavailable` because replay mode/status is not native/comparable: `oracle-replay-adapted`. | Confirms no parity comparison should be reported. |
| [Issue #1467 replay metrics](issue_1467_carla_replay_metrics.md) | A generated native smoke probe emitted comparable `success`, `collision`, and `intervention_rate`. | Shows the metric path is worth preserving after fixture alignment repair. |

## Outcome Classification

- `pause_until_carla_host`: rejected. #1509 records Docker preflight availability and a live
  replay against `carlasim/carla:0.9.16`; the blocker is not host access.
- `stop_current_parity_lane`: rejected. #1467 shows the native metric-emission path can work on a
  generated smoke probe, so stopping the lane would discard a viable technical path.
- `proceed_to_alignment_repair`: selected. Existing certified fixture evidence fails closed because
  CARLA applied material spawn projection. The next useful work is fixture or transform repair, not
  broader campaign execution.

## Gated Follow-Up

Issue #1510 and Issue #1511 should remain blocked unless a new compact bundle proves all of the
following:

- replay mode is `native` or explicitly `aligned`;
- native mode has zero unplanned projection, or aligned mode has a pre-declared reversible transform
  and threshold;
- replay status is `oracle-replay`;
- comparable Robot-SF and CARLA metrics are emitted;
- fixture provenance is durable and tied to the certified scenario contract.

## Claim Boundary

This note is CARLA lane routing only. It does not establish replay parity, simulator-transfer
validity, benchmark-strength CARLA evidence, or paper-facing CARLA claims. The adapted replay is
negative gating evidence: useful for deciding the next repair, but not success evidence.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_2276_carla_parity_lane_decision_2026-06-05/summary.json
python -m json.tool docs/context/evidence/issue_1508_carla_eligibility_2026-05-31/candidate_eligibility_summary.json
python -m json.tool docs/context/evidence/issue_1509_carla_native_fixture_2026-05-31/live_replay_pr_promoted.json
python -m json.tool docs/context/evidence/issue_1509_carla_native_fixture_2026-05-31/parity_report_pr_promoted.json
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
