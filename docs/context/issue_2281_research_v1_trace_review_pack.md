# Issue #2281 Research-v1 Trace Review Pack

Issue: [#2281](https://github.com/ll7/robot_sf_ll7/issues/2281)
Parent issue: [#2159](https://github.com/ll7/robot_sf_ll7/issues/2159)
Inputs: [issue_2269_research_v1_trace_case_selection.md](issue_2269_research_v1_trace_case_selection.md),
[issue_2280_research_v1_first_trace_review.md](issue_2280_research_v1_first_trace_review.md)
Date: 2026-06-05
Status: diagnostic synthesis pack; not benchmark or paper-facing evidence.

## Goal

Assemble the selected research-v1 failure-case trace-review cases into one compact diagnostic pack.
This pack converts the #2269 case-selection manifest and the #2280 first reviewed case into a
single status surface for the #2159 why-first review lane.

Compact synthesis summary:
`docs/context/evidence/issue_2281_trace_review_pack_2026-06-05/summary.json`.

## Pack Status

| Case | Status | Mechanism signal | Verdict | Main caveat |
| --- | --- | --- | --- | --- |
| `ammv_head_on_corridor_mechanism_activation` | blocked | Direct AMMV force term activated in a mechanism probe. | `gather_more_evidence` | No matched `simulation_trace_export.v1` traces or benchmark-row AMMV metadata. |
| `amv_cross_trap_hazard_odd_slice` | blocked | Cross-trap AMV slice has hazard/ODD coverage evidence. | `gather_more_evidence` | No selected per-planner trace slices or annotation set. |
| `head_on_corridor_route_offset_response` | reviewed | Route offset changed closest-approach geometry; hybrid seed `117` shifted closest approach earlier and farther along route. | `diagnostic_supported_case` | Terminal outcomes stayed `max_steps`; command-source telemetry missing. |
| `leave_group_speed_outcome_flip` | reviewed | ORCA seed `258` changed from collision to success with a small clearance/phase shift. | `diagnostic_supported_case` | Effect is fragile and seed-local, not a robust speed-perturbation claim. |
| `intersection_wait_speed_p050_phase_response` | reviewed | `+0.5 m/s` pedestrian speed offset repeated a strong negative clearance shift across 9 pairs. | `diagnostic_supported_case` | Same `max_steps` terminal outcome; pedestrian IDs are index-based. |

The pack satisfies the #2281 stop condition with three durable reviewed cases and two explicit
missing-input blockers.

## Synthesis Table

| Mechanism | Source issue | Evidence tier | Config | Seeds | Artifacts | Metrics | Verdict | Caveats |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| AMMV Social Force direct activation | #2168 / #2269 | diagnostic mechanism probe | `classic_head_on_corridor_low` direct probe | `111`, `112`, `113` | `docs/context/evidence/issue_2168_ammv_social_force_pair_2026-06-03/summary.json` | AMMV force magnitude and intrusion count | `blocked_for_pack_render` | Adapter rows did not surface AMMV metadata; matched renderable traces absent. |
| AMV cross-trap hazard/ODD coverage | #2164 / #2269 | diagnostic coverage slice | #1484 cross-kinematics reports | `111` | `docs/context/evidence/issue_2164_amv_hazard_odd_join_2026-06-03/hazard_odd_coverage_summary.json` | hazard, ODD, scenario-contract coverage | `blocked_for_pack_render` | Coverage evidence is not a trace review; selected trace slices absent. |
| Head-on corridor pedestrian route offset | #1939 / #2280 | compact trace-slice review | `configs/scenarios/perturbations/issue_1610_ped_route_offset_pilot_v1.yaml` | `111`, `112`, `116`, `117` | `docs/context/evidence/issue_1939_corridor_trace_response_2026-05-31/closest_approach_trace_slices.json` | 12 completed pairs; mean clearance delta `+0.153489 m`; hybrid seed `117` progress delta `+6.581509 m` | `diagnostic_supported_case` | Geometry/progress signal only; terminal `max_steps` unchanged and command-source timeline absent. |
| ORCA leave-group speed outcome flip | #1945 | compact trace-slice review | `configs/scenarios/perturbations/issue_1610_ped_speed_pilot_v1.yaml` | `258`, `259`, `260` | `docs/context/evidence/issue_1945_orca_leave_group_speed_trace_2026-06-01/closest_approach_trace_slices.json` | 3 completed pairs; seed `258` collision-to-success; mean clearance delta `+0.034504 m` | `diagnostic_supported_case` | Seed-local phase/order mechanism; neighboring seeds do not show broad outcome improvement. |
| Intersection-wait speed `p050` phase response | #1953 | compact trace-slice review | `configs/scenarios/perturbations/issue_1610_intersection_wait_phase_grid_v1.yaml` plus ignored filtered target manifest | `240`, `241`, `242` | `docs/context/evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/closest_approach_trace_slices_speed_h1_p050.json` | 9 completed pairs; mean clearance delta `-3.862581 m`; same max-steps termination | `diagnostic_supported_case` | Phase/clearance sensitivity only; no terminal outcome change and no stable pedestrian IDs. |

## Recommendation

Recommendation: `gather_more_evidence`.

The current pack is valuable enough to continue the why-first trace-review lane, but it should not
be promoted into benchmark-strength or paper-facing evidence. The next useful step is not another
rubric. It is either:

- export AMV-specific renderable traces for `ammv_head_on_corridor_mechanism_activation`, or
- build a small annotation set for one of the three already reviewed compact trace-slice cases.

Do not broaden to a viewer/UI project until at least one AMV-specific case has durable renderable
traces or a maintained annotation contract.

## Claim Boundary

This pack is diagnostic synthesis only. It explains where selected mechanisms activated, changed
local geometry, changed terminal outcome, or failed to preserve the necessary telemetry. It does
not establish AMV transfer, calibrated actuation, broad planner robustness, or paper-facing
failure-analysis claims.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_2281_trace_review_pack_2026-06-05/summary.json
python -m json.tool docs/context/evidence/issue_1939_corridor_trace_response_2026-05-31/closest_approach_trace_slices.json
python -m json.tool docs/context/evidence/issue_1945_orca_leave_group_speed_trace_2026-06-01/closest_approach_trace_slices.json
python -m json.tool docs/context/evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/closest_approach_trace_slices_speed_h1_p050.json
python - <<'PY'
from pathlib import Path
import yaml
data = yaml.safe_load(Path("docs/context/evidence/issue_2269_research_v1_trace_case_selection_2026-06-05/case_selection_manifest.yaml").read_text())
assert len(data["cases"]) == 5
assert sum(1 for case in data["cases"] if case["durable_case_ready"]) == 3
for case in data["cases"]:
    for path in case["source_evidence"]:
        assert Path(path).exists(), path
PY
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
