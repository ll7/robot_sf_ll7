# Nominal-Sanity Leaderboard

This page starts with rows whose nominal-sanity evidence is already represented by tracked
policy-search reports. Nominal-sanity rows are useful triage evidence; they do not by themselves
promote a planner to a paper-facing benchmark result.

| planner | suite | success | collision | near_miss | low_progress | min_distance | runtime | benchmark_track | evidence_uri | status | claim_boundary |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| `ppo_issue791_best_v1` | `policy_search:nominal_sanity` | `0.2778` | `0.0000` | `0.2222` | `10/18` | `3.7136` | `not_recorded` | `policy_search_nominal_sanity` | [`docs/context/policy_search/reports/2026-05-05_ppo_issue791_best_v1_nominal_sanity.md`](../context/policy_search/reports/2026-05-05_ppo_issue791_best_v1_nominal_sanity.md) | `revise` | Strongest learned-only nominal-sanity row in that pass, but still limited by low-progress timeouts and intrusive near misses. |

Additional nominal-sanity candidates should be added only after their tracked reports or compact
summaries are checked against this row contract.
