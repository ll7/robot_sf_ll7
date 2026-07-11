<!-- AI-GENERATED NEEDS-REVIEW -->

# Retained Campaign Flat vs Hierarchical Interval Comparison (issue #5139)

**Claim boundary:** diagnostic-only, analysis-only: flat versus hierarchical interval widths on the retained issue #1454 exploratory campaign bundle. This post-hoc reuse was not pre-registered for issue #5139 and does not establish benchmark, planner-ranking, paper, or dissertation claims.

**Evidence status:** `diagnostic-only (analysis-only)`

**Major caveat:** this is a post-hoc analysis of an exploratory retained campaign, not the pre-registered successor campaign and not benchmark-strength evidence.

## Plain-language summary

This report compares confidence-interval widths when retained campaign episodes are treated as independent versus when their scenario grouping is respected. It exercises the merged scenario-hierarchical bootstrap and cluster-robust binary intervals on real campaign rows while keeping the result analysis-only.

## Deterministic provenance

- campaign: `issue1454-s10-h500-candidates`
- episode table: `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/reports/seed_episode_rows.csv`
- episode table SHA-256: `fd758fceb72cecdd375b5f59b7e9bd8fa639cbb5c74c35db362fb751a9cada8f`
- campaign manifest: `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/campaign_manifest.json`
- campaign manifest SHA-256: `073f42e05e92bb7a143c5fae42286f455d72a0696369142452618a138fa99408`
- campaign commit: `4941ac48f1f4e65053bbfcbbc94a55a336fad9ea`
- config hash: `b5a3a8b80493c728`
- rows: 5760
- planners: 12
- scenario cells: 48
- seeds: 10
- analysis groups: `(planner_key, kinematics)`; hierarchical cluster: `scenario_id`
- bootstrap samples: 1000
- confidence: 0.95
- master seed: 5139

## Interval-width ratios

A ratio above 1 means the scenario-hierarchical interval is wider than the flat interval on the same retained rows. Binary endpoints compare cluster-robust intervals against flat Wilson intervals.

| planner | kinematics | metric | flat width | hierarchical width | ratio |
| --- | --- | --- | ---: | ---: | ---: |
| goal | differential_drive | collision_rate | 0.086838 | 0.238801 | 2.750 |
| goal | differential_drive | near_miss | 2.631250 | 6.462500 | 2.456 |
| goal | differential_drive | snqi | 0.028030 | 0.071558 | 2.553 |
| goal | differential_drive | success_rate | 0.038734 | 0.076525 | 1.976 |
| goal | differential_drive | time_to_goal | 0.008370 | 0.014197 | 1.696 |
| hybrid_rule_v3_fast_progress | differential_drive | collision_rate | 0.030906 | 0.046625 | 1.509 |
| hybrid_rule_v3_fast_progress | differential_drive | near_miss | 6.929167 | 16.383333 | 2.364 |
| hybrid_rule_v3_fast_progress | differential_drive | snqi | 0.042679 | 0.108988 | 2.554 |
| hybrid_rule_v3_fast_progress | differential_drive | success_rate | 0.073043 | 0.170189 | 2.330 |
| hybrid_rule_v3_fast_progress | differential_drive | time_to_goal | 0.030427 | 0.070168 | 2.306 |
| hybrid_rule_v3_fast_progress_static_escape | differential_drive | collision_rate | 0.033754 | 0.047395 | 1.404 |
| hybrid_rule_v3_fast_progress_static_escape | differential_drive | near_miss | 6.827083 | 17.010417 | 2.492 |
| hybrid_rule_v3_fast_progress_static_escape | differential_drive | snqi | 0.043688 | 0.109489 | 2.506 |
| hybrid_rule_v3_fast_progress_static_escape | differential_drive | success_rate | 0.061251 | 0.123788 | 2.021 |
| hybrid_rule_v3_fast_progress_static_escape | differential_drive | time_to_goal | 0.028634 | 0.071027 | 2.480 |
| hybrid_rule_v3_fast_progress_static_escape_continuous | differential_drive | collision_rate | 0.028827 | 0.044443 | 1.542 |
| hybrid_rule_v3_fast_progress_static_escape_continuous | differential_drive | near_miss | 7.400000 | 18.064583 | 2.441 |
| hybrid_rule_v3_fast_progress_static_escape_continuous | differential_drive | snqi | 0.045420 | 0.118448 | 2.608 |
| hybrid_rule_v3_fast_progress_static_escape_continuous | differential_drive | success_rate | 0.058819 | 0.114499 | 1.947 |
| hybrid_rule_v3_fast_progress_static_escape_continuous | differential_drive | time_to_goal | 0.028609 | 0.066744 | 2.333 |
| orca | differential_drive | collision_rate | 0.065623 | 0.136979 | 2.087 |
| orca | differential_drive | near_miss | 4.229167 | 9.893750 | 2.339 |
| orca | differential_drive | snqi | 0.082217 | 0.246925 | 3.003 |
| orca | differential_drive | success_rate | 0.074544 | 0.173312 | 2.325 |
| orca | differential_drive | time_to_goal | 0.036320 | 0.086996 | 2.395 |
| ppo | differential_drive | collision_rate | 0.070602 | 0.182838 | 2.590 |
| ppo | differential_drive | near_miss | 2.168750 | 4.387500 | 2.023 |
| ppo | differential_drive | snqi | 0.055199 | 0.151637 | 2.747 |
| ppo | differential_drive | success_rate | 0.073300 | 0.191422 | 2.611 |
| ppo | differential_drive | time_to_goal | 0.041619 | 0.101150 | 2.430 |
| prediction_planner | differential_drive | collision_rate | 0.087318 | 0.182313 | 2.088 |
| prediction_planner | differential_drive | near_miss | 5.720833 | 13.410417 | 2.344 |
| prediction_planner | differential_drive | snqi | 0.041286 | 0.111378 | 2.698 |
| prediction_planner | differential_drive | success_rate | 0.088931 | 0.190411 | 2.141 |
| prediction_planner | differential_drive | time_to_goal | 0.032915 | 0.069616 | 2.115 |
| sacadrl | differential_drive | collision_rate | 0.084432 | 0.236253 | 2.798 |
| sacadrl | differential_drive | near_miss | 2.106250 | 4.722917 | 2.242 |
| sacadrl | differential_drive | snqi | 0.041079 | 0.105885 | 2.578 |
| sacadrl | differential_drive | success_rate | 0.045609 | 0.117966 | 2.586 |
| sacadrl | differential_drive | time_to_goal | 0.018882 | 0.049184 | 2.605 |
| scenario_adaptive_hybrid_orca_v1 | differential_drive | collision_rate | 0.032836 | 0.047169 | 1.437 |
| scenario_adaptive_hybrid_orca_v1 | differential_drive | near_miss | 6.393750 | 15.777083 | 2.468 |
| scenario_adaptive_hybrid_orca_v1 | differential_drive | snqi | 0.041487 | 0.099119 | 2.389 |
| scenario_adaptive_hybrid_orca_v1 | differential_drive | success_rate | 0.059650 | 0.122266 | 2.050 |
| scenario_adaptive_hybrid_orca_v1 | differential_drive | time_to_goal | 0.030114 | 0.069280 | 2.301 |
| scenario_adaptive_hybrid_orca_v2_collision_guard | differential_drive | collision_rate | 0.032836 | 0.047169 | 1.437 |
| scenario_adaptive_hybrid_orca_v2_collision_guard | differential_drive | near_miss | 6.575000 | 15.260417 | 2.321 |
| scenario_adaptive_hybrid_orca_v2_collision_guard | differential_drive | snqi | 0.042283 | 0.104054 | 2.461 |
| scenario_adaptive_hybrid_orca_v2_collision_guard | differential_drive | success_rate | 0.059650 | 0.122266 | 2.050 |
| scenario_adaptive_hybrid_orca_v2_collision_guard | differential_drive | time_to_goal | 0.027453 | 0.067399 | 2.455 |
| social_force | differential_drive | collision_rate | 0.086752 | 0.239474 | 2.760 |
| social_force | differential_drive | near_miss | 1.510417 | 3.383333 | 2.240 |
| social_force | differential_drive | snqi | 0.054185 | 0.147791 | 2.728 |
| social_force | differential_drive | success_rate | 0.024070 | 0.035426 | 1.472 |
| social_force | differential_drive | time_to_goal | 0.001782 | 0.003410 | 1.914 |
| socnav_sampling | differential_drive | collision_rate | 0.087318 | 0.222064 | 2.543 |
| socnav_sampling | differential_drive | near_miss | 0.612500 | 1.410417 | 2.303 |
| socnav_sampling | differential_drive | snqi | 0.041248 | 0.113011 | 2.740 |
| socnav_sampling | differential_drive | success_rate | 0.087318 | 0.222064 | 2.543 |
| socnav_sampling | differential_drive | time_to_goal | 0.039073 | 0.104317 | 2.670 |

## Bounded summary

### Binary rate metrics

- comparisons: 24
- minimum ratio: 1.404
- median ratio: 2.088
- mean ratio: 2.125
- maximum ratio: 2.798

### Non-rate metrics

- comparisons: 36
- minimum ratio: 1.696
- median ratio: 2.448
- mean ratio: 2.425
- maximum ratio: 3.003

Ratios describe this retained exploratory bundle only. A ratio above one means the scenario-hierarchical interval is wider; a ratio at or below one is retained rather than filtered and does not invalidate the method.

## Reproduce

```bash
uv run python scripts/analysis/compare_flat_vs_hierarchical_intervals_issue_5139.py \
  --retained-bundle docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/reports/seed_episode_rows.csv \
  --retained-manifest docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/campaign_manifest.json
```
