# Smoke Leaderboard

This page lists smoke rows with durable tracked evidence. Smoke success proves that a narrow
execution path ran; it is not benchmark-strength planner ranking evidence.

| planner | suite | success | collision | near_miss | low_progress | min_distance | runtime | benchmark_track | evidence_uri | status | claim_boundary |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| `orca_residual_guarded_ppo_v0` | `policy_search:smoke` | `1.0000` | `0.0000` | `0.0000` | `0` | `not_recorded` | `16.2991s` | `policy_search_smoke` | [`docs/context/evidence/issue_1428_orca_residual_lineage_2026-05-24/orca_residual_guarded_ppo_v0_smoke_summary.json`](../context/evidence/issue_1428_orca_residual_lineage_2026-05-24/orca_residual_guarded_ppo_v0_smoke_summary.json) | `pass` | Runtime smoke only; not learned-residual training evidence. |
| `ppo_issue791_best_v1` | `policy_search:smoke` | `1.0000` | `0.0000` | `0.0000` | `not_recorded` | `not_recorded` | `not_recorded` | `policy_search_smoke` | [`docs/context/policy_search/reports/2026-05-05_best_learning_policy.md`](../context/policy_search/reports/2026-05-05_best_learning_policy.md) | `pass` | Smoke proves artifact hydration and runnable learned-policy inference only. |
| `hybrid_orca_sampler_v1` | `policy_search:smoke` | `1.0000` | `0.0000` | `0.0000` | `0` | `not_recorded` | `not_recorded` | `policy_search_smoke` | [`docs/context/policy_search/reports/2026-05-31_hybrid_orca_sampler_v1_smoke.md`](../context/policy_search/reports/2026-05-31_hybrid_orca_sampler_v1_smoke.md) | `pass` | Smoke evidence for the 120-step horizon probe only; not candidate promotion. |

Rows omitted from this first page are not negative results. They are simply not yet audited into
this static row contract.
