# AMV Actuation Leaderboard

This page covers synthetic AMV actuation surfaces. These rows are diagnostic unless a linked
claim-map decision explicitly promotes them. Interpret every row through the
[AMV actuation evidence ladder](../context/issue_2230_amv_actuation_evidence_ladder.md): synthetic
diagnostics are not platform-class proxy evidence or hardware-calibrated AMV evidence. The first
populated rows come from compact tracked evidence for Issue #1569, whose own summary says the smoke
does not strengthen AMV performance claims.

| planner | suite | success | collision | near_miss | low_progress | min_distance | runtime | benchmark_track | evidence_uri | status | claim_boundary |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| `goal` | `amv_actuation_smoke` | `0.0000` | `0.2000` | `1.0000` | `not_recorded` | `not_recorded` | `13.4813s` | `synthetic_amv_actuation_smoke` | [`docs/context/evidence/issue_1569_amv_actuation_smoke_2026-05-27/summary.json`](../context/evidence/issue_1569_amv_actuation_smoke_2026-05-27/summary.json) | `successful_evidence` | Executable synthetic diagnostic row; no paper-facing AMV performance claim. |
| `orca` | `amv_actuation_smoke` | `0.0000` | `0.1333` | `3.4667` | `not_recorded` | `not_recorded` | `7.9955s` | `synthetic_amv_actuation_smoke` | [`docs/context/evidence/issue_1569_amv_actuation_smoke_2026-05-27/summary.json`](../context/evidence/issue_1569_amv_actuation_smoke_2026-05-27/summary.json) | `successful_evidence` | Executable synthetic diagnostic row; ORCA projection diagnostics remain caveated. |
| `social_force` | `amv_actuation_smoke` | `0.0000` | `0.2667` | `1.0000` | `not_recorded` | `not_recorded` | `6.4261s` | `synthetic_amv_actuation_smoke` | [`docs/context/evidence/issue_1569_amv_actuation_smoke_2026-05-27/summary.json`](../context/evidence/issue_1569_amv_actuation_smoke_2026-05-27/summary.json) | `successful_evidence` | Executable synthetic diagnostic row; command-space and projection-policy metadata remain caveated. |
| `actuation_aware_hybrid_rule_v0` | `amv_actuation_smoke` | `0.0000` | `0.0000` | `0.0000` | `timeout_low_progress: 1` | `2.3627` | `12.0865s` | `synthetic_amv_actuation_smoke` | [`docs/context/policy_search/reports/2026-05-31_actuation_aware_hybrid_rule_v0_amv_actuation_smoke.md`](../context/policy_search/reports/2026-05-31_actuation_aware_hybrid_rule_v0_amv_actuation_smoke.md) | `successful_evidence` | Executable synthetic diagnostic row; no paper-facing AMV performance claim, and low-progress timeout remains a planner limitation. |
| `hybrid_rule_v3_fast_progress` | `amv_actuation_smoke` | `0.0000` | `0.0000` | `0.0000` | `timeout_low_progress: 1` | `2.1571` | `not_recorded` | `synthetic_amv_actuation_smoke` | [`docs/context/evidence/issue_2224_amv_actuation_ranking_2026-06-04/comparison.json`](../context/evidence/issue_2224_amv_actuation_ranking_2026-06-04/comparison.json) | `diagnostic_direction_only` | Matched smoke row for issue #2224; command clip `0.2750`, yaw saturation `0.0000`; no planner-ranking claim. |
| `actuation_aware_hybrid_rule_v0` | `amv_actuation_smoke` | `0.0000` | `0.0000` | `0.0000` | `timeout_low_progress: 1` | `2.3627` | `not_recorded` | `synthetic_amv_actuation_smoke` | [`docs/context/evidence/issue_2224_amv_actuation_ranking_2026-06-04/comparison.json`](../context/evidence/issue_2224_amv_actuation_ranking_2026-06-04/comparison.json) | `diagnostic_direction_only` | Matched smoke row for issue #2224; command clip `0.1875`, yaw saturation `0.0000`; both compared candidates timed out. |

The Issue #1569 bundle reports AMV coverage `warn`, not a clean AMV-coverage pass. Keep that caveat
with any downstream reuse of these rows.

The Issue #2224 comparison is a one-episode diagnostic-only row. It suggests lower synthetic
command clipping for the actuation-aware scorer, but it does not establish material ranking change
because both compared candidates timed out with zero success.
