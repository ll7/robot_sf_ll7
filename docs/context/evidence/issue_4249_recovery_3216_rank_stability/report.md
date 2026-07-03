# Headline 7x7 CI + Rank-Stability Report (#3216)

- **Classification**: `blocked_until_run`
- **Rationale**: Per-cell seed budget reaches paper-grade threshold (min_seeds=20 >= 20), but paper-grade promotion requires the predeclared S20/S30 SLURM headline run (#1554) and claim-card review. This local harness emits the statistics only.
- **Claim boundary**: Per-cell confidence intervals and rank-stability are reported with explicit fail-closed cell status. This harness makes NO paper-grade or planner-ranking claim on its own: the paper-grade 7x7 headline run requires the increased seed budget (S20/S30 via #1554) and is SLURM. On insufficient seed budget the result is classified blocked_until_run or diagnostic.
- **git HEAD**: `3e557e47af2c35398e38669f2bffb7f692ab06b8`
- **Cells**: 315 counted / 0 excluded (of 315 rows)
- **Expected grid**: 9 observed / 9 expected; 0 missing; 0 unexpected
- **Rank metric**: `snqi`
- **Rank profile**: `snqi_diagnostic`
- **Rank metric contract**: `invalid` - SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247

## Canonical owners reused (not reinvented)

- `robot_sf.benchmark.seed_variance._stats_for_vals`
- `robot_sf.benchmark.seed_variance._bootstrap_mean_ci`
- `robot_sf.benchmark.fidelity_rank_stability.rank_planners`
- `robot_sf.benchmark.fidelity_rank_stability.kendall_tau`
- `robot_sf.benchmark.fidelity_rank_stability.count_rank_flips`
- `robot_sf.benchmark.canonical_table_export.load_rows_json`

## Per-cell confidence intervals (counted cells)

| scenario | planner | status | seeds | metric | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| accompanying_peer | goal | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | goal | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | goal | ok | 20 | snqi | -0.1052 | -0.1130 | -0.0984 |
| accompanying_peer | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | 0.1060 | 0.0979 | 0.1130 |
| accompanying_peer | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| accompanying_peer | orca | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | orca | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | orca | ok | 20 | snqi | 0.1122 | 0.1097 | 0.1144 |
| accompanying_peer | orca | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| accompanying_peer | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | ppo | ok | 20 | near_misses | 0.4000 | 0.0000 | 0.9500 |
| accompanying_peer | ppo | ok | 20 | snqi | 0.0835 | 0.0694 | 0.0967 |
| accompanying_peer | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| accompanying_peer | prediction_planner | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| accompanying_peer | prediction_planner | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | prediction_planner | ok | 20 | snqi | -0.0864 | -0.1108 | -0.0538 |
| accompanying_peer | prediction_planner | ok | 20 | success | 0.1000 | 0.0000 | 0.2500 |
| accompanying_peer | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| accompanying_peer | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | sacadrl | ok | 20 | snqi | -0.6550 | -0.7160 | -0.5927 |
| accompanying_peer | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | 0.1060 | 0.0979 | 0.1130 |
| accompanying_peer | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| accompanying_peer | social_force | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| accompanying_peer | social_force | ok | 20 | near_misses | 3.8500 | 2.8000 | 5.7500 |
| accompanying_peer | social_force | ok | 20 | snqi | -0.8648 | -0.9149 | -0.8340 |
| accompanying_peer | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | socnav_sampling | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| accompanying_peer | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| accompanying_peer | socnav_sampling | ok | 20 | snqi | -0.3031 | -0.3389 | -0.2680 |
| accompanying_peer | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| blind_corner | goal | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| blind_corner | goal | ok | 20 | near_misses | 6.1000 | 6.0000 | 6.2500 |
| blind_corner | goal | ok | 20 | snqi | -0.3665 | -0.3790 | -0.3541 |
| blind_corner | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| blind_corner | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.9000 | 0.7500 | 1.0000 |
| blind_corner | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 2.9000 | 2.4000 | 3.3000 |
| blind_corner | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.3096 | -0.3367 | -0.2738 |
| blind_corner | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| blind_corner | orca | ok | 20 | collisions | 0.7500 | 0.5500 | 0.9000 |
| blind_corner | orca | ok | 20 | near_misses | 4.1000 | 1.6000 | 7.4000 |
| blind_corner | orca | ok | 20 | snqi | -0.2989 | -0.3230 | -0.2754 |
| blind_corner | orca | ok | 20 | success | 0.2500 | 0.1000 | 0.4500 |
| blind_corner | ppo | ok | 20 | collisions | 0.2500 | 0.1000 | 0.4500 |
| blind_corner | ppo | ok | 20 | near_misses | 5.9500 | 4.3500 | 7.2500 |
| blind_corner | ppo | ok | 20 | snqi | -0.1501 | -0.2007 | -0.1096 |
| blind_corner | ppo | ok | 20 | success | 0.7500 | 0.5500 | 0.9000 |
| blind_corner | prediction_planner | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| blind_corner | prediction_planner | ok | 20 | near_misses | 2.0000 | 1.0500 | 3.0000 |
| blind_corner | prediction_planner | ok | 20 | snqi | -0.2622 | -0.2896 | -0.2375 |
| blind_corner | prediction_planner | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| blind_corner | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| blind_corner | sacadrl | ok | 20 | near_misses | 9.8000 | 7.4500 | 11.9000 |
| blind_corner | sacadrl | ok | 20 | snqi | -0.5949 | -0.6536 | -0.5278 |
| blind_corner | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| blind_corner | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.9000 | 0.7500 | 1.0000 |
| blind_corner | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 2.9000 | 2.4000 | 3.3000 |
| blind_corner | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.3096 | -0.3367 | -0.2738 |
| blind_corner | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| blind_corner | social_force | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| blind_corner | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| blind_corner | social_force | ok | 20 | snqi | -0.6758 | -0.7113 | -0.6337 |
| blind_corner | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| blind_corner | socnav_sampling | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| blind_corner | socnav_sampling | ok | 20 | near_misses | 2.5500 | 2.3500 | 2.7500 |
| blind_corner | socnav_sampling | ok | 20 | snqi | -0.2857 | -0.2946 | -0.2765 |
| blind_corner | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| bottleneck | goal | ok | 20 | collisions | 0.7000 | 0.6500 | 0.7500 |
| bottleneck | goal | ok | 20 | near_misses | 12.2750 | 8.9369 | 15.7384 |
| bottleneck | goal | ok | 20 | snqi | -0.2743 | -0.2889 | -0.2612 |
| bottleneck | goal | ok | 20 | success | 0.0625 | 0.0125 | 0.1125 |
| bottleneck | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0750 | 0.0250 | 0.1250 |
| bottleneck | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 33.6750 | 27.3238 | 40.3503 |
| bottleneck | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.1460 | -0.1587 | -0.1336 |
| bottleneck | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.9250 | 0.8750 | 0.9750 |
| bottleneck | orca | ok | 20 | collisions | 0.2875 | 0.2250 | 0.3625 |
| bottleneck | orca | ok | 20 | near_misses | 24.7250 | 21.5500 | 28.0753 |
| bottleneck | orca | ok | 20 | snqi | -0.5289 | -0.5764 | -0.4812 |
| bottleneck | orca | ok | 20 | success | 0.6625 | 0.5875 | 0.7375 |
| bottleneck | ppo | ok | 20 | collisions | 0.7500 | 0.7125 | 0.7875 |
| bottleneck | ppo | ok | 20 | near_misses | 7.6000 | 7.1375 | 8.0253 |
| bottleneck | ppo | ok | 20 | snqi | -0.4675 | -0.4909 | -0.4434 |
| bottleneck | ppo | ok | 20 | success | 0.2625 | 0.2500 | 0.2875 |
| bottleneck | prediction_planner | ok | 20 | collisions | 0.4875 | 0.4000 | 0.5750 |
| bottleneck | prediction_planner | ok | 20 | near_misses | 37.0625 | 28.7622 | 45.6528 |
| bottleneck | prediction_planner | ok | 20 | snqi | -0.1992 | -0.2317 | -0.1731 |
| bottleneck | prediction_planner | ok | 20 | success | 0.5125 | 0.4250 | 0.6000 |
| bottleneck | sacadrl | ok | 20 | collisions | 0.7625 | 0.7500 | 0.7875 |
| bottleneck | sacadrl | ok | 20 | near_misses | 4.2625 | 3.2116 | 5.2753 |
| bottleneck | sacadrl | ok | 20 | snqi | -0.4105 | -0.4605 | -0.3649 |
| bottleneck | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| bottleneck | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0750 | 0.0250 | 0.1250 |
| bottleneck | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 33.6750 | 27.3238 | 40.3503 |
| bottleneck | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.1460 | -0.1587 | -0.1336 |
| bottleneck | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.9250 | 0.8750 | 0.9750 |
| bottleneck | social_force | ok | 20 | collisions | 0.0625 | 0.0250 | 0.1125 |
| bottleneck | social_force | ok | 20 | near_misses | 1.1875 | 0.7500 | 1.6250 |
| bottleneck | social_force | ok | 20 | snqi | -0.9265 | -0.9539 | -0.9016 |
| bottleneck | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| bottleneck | socnav_sampling | ok | 20 | collisions | 0.7625 | 0.7500 | 0.7875 |
| bottleneck | socnav_sampling | ok | 20 | near_misses | 3.3000 | 2.8000 | 3.8500 |
| bottleneck | socnav_sampling | ok | 20 | snqi | -0.2105 | -0.2323 | -0.1931 |
| bottleneck | socnav_sampling | ok | 20 | success | 0.2375 | 0.2125 | 0.2500 |
| circular_crossing | goal | ok | 20 | collisions | 0.2000 | 0.0500 | 0.4000 |
| circular_crossing | goal | ok | 20 | near_misses | 0.3500 | 0.0000 | 1.0000 |
| circular_crossing | goal | ok | 20 | snqi | -0.1629 | -0.2038 | -0.1274 |
| circular_crossing | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| circular_crossing | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.2000 | 0.0500 | 0.4000 |
| circular_crossing | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 0.3500 | 0.0000 | 1.0000 |
| circular_crossing | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | 0.0080 | -0.0731 | 0.0763 |
| circular_crossing | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.8000 | 0.6000 | 0.9500 |
| circular_crossing | orca | ok | 20 | collisions | 0.2000 | 0.0500 | 0.4000 |
| circular_crossing | orca | ok | 20 | near_misses | 0.3000 | 0.0000 | 0.8500 |
| circular_crossing | orca | ok | 20 | snqi | 0.0074 | -0.0726 | 0.0759 |
| circular_crossing | orca | ok | 20 | success | 0.8000 | 0.6000 | 0.9500 |
| circular_crossing | ppo | ok | 20 | collisions | 0.2000 | 0.0500 | 0.4000 |
| circular_crossing | ppo | ok | 20 | near_misses | 0.3000 | 0.0000 | 0.8500 |
| circular_crossing | ppo | ok | 20 | snqi | -0.0813 | -0.1430 | -0.0229 |
| circular_crossing | ppo | ok | 20 | success | 0.8000 | 0.6000 | 0.9500 |
| circular_crossing | prediction_planner | ok | 20 | collisions | 0.2000 | 0.0500 | 0.4000 |
| circular_crossing | prediction_planner | ok | 20 | near_misses | 0.3500 | 0.0000 | 1.0000 |
| circular_crossing | prediction_planner | ok | 20 | snqi | 0.0095 | -0.0713 | 0.0779 |
| circular_crossing | prediction_planner | ok | 20 | success | 0.8000 | 0.6000 | 0.9500 |
| circular_crossing | sacadrl | ok | 20 | collisions | 0.2000 | 0.0500 | 0.4000 |
| circular_crossing | sacadrl | ok | 20 | near_misses | 0.3500 | 0.0000 | 1.0000 |
| circular_crossing | sacadrl | ok | 20 | snqi | -0.2397 | -0.2838 | -0.1995 |
| circular_crossing | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| circular_crossing | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.2000 | 0.0500 | 0.4000 |
| circular_crossing | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 0.3500 | 0.0000 | 1.0000 |
| circular_crossing | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | 0.0080 | -0.0731 | 0.0763 |
| circular_crossing | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.8000 | 0.6000 | 0.9500 |
| circular_crossing | social_force | ok | 20 | collisions | 0.2000 | 0.0500 | 0.4000 |
| circular_crossing | social_force | ok | 20 | near_misses | 0.3000 | 0.0000 | 0.8500 |
| circular_crossing | social_force | ok | 20 | snqi | -0.3562 | -0.3758 | -0.3358 |
| circular_crossing | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| circular_crossing | socnav_sampling | ok | 20 | collisions | 0.2000 | 0.0500 | 0.4000 |
| circular_crossing | socnav_sampling | ok | 20 | near_misses | 0.3000 | 0.0000 | 0.8500 |
| circular_crossing | socnav_sampling | ok | 20 | snqi | -0.0233 | -0.0998 | 0.0439 |
| circular_crossing | socnav_sampling | ok | 20 | success | 0.8000 | 0.6000 | 0.9500 |
| cross_trap | goal | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| cross_trap | goal | ok | 20 | near_misses | 11.4500 | 8.0329 | 15.0671 |
| cross_trap | goal | ok | 20 | snqi | -0.4113 | -0.4513 | -0.3696 |
| cross_trap | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| cross_trap | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.4333 | 0.2833 | 0.5833 |
| cross_trap | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 16.4000 | 9.3329 | 24.6000 |
| cross_trap | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.3192 | -0.4343 | -0.2152 |
| cross_trap | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.4833 | 0.3333 | 0.6333 |
| cross_trap | orca | ok | 20 | collisions | 0.2833 | 0.1500 | 0.4167 |
| cross_trap | orca | ok | 20 | near_misses | 27.0667 | 19.5500 | 34.3350 |
| cross_trap | orca | ok | 20 | snqi | -0.3567 | -0.4451 | -0.2714 |
| cross_trap | orca | ok | 20 | success | 0.6833 | 0.5500 | 0.8167 |
| cross_trap | ppo | ok | 20 | collisions | 0.2000 | 0.1167 | 0.3000 |
| cross_trap | ppo | ok | 20 | near_misses | 9.9667 | 7.6167 | 12.5167 |
| cross_trap | ppo | ok | 20 | snqi | -0.2483 | -0.3109 | -0.1956 |
| cross_trap | ppo | ok | 20 | success | 0.7833 | 0.6833 | 0.8833 |
| cross_trap | prediction_planner | ok | 20 | collisions | 0.7333 | 0.6000 | 0.8500 |
| cross_trap | prediction_planner | ok | 20 | near_misses | 34.3167 | 21.3000 | 48.7675 |
| cross_trap | prediction_planner | ok | 20 | snqi | -0.4073 | -0.4845 | -0.3256 |
| cross_trap | prediction_planner | ok | 20 | success | 0.2667 | 0.1500 | 0.4000 |
| cross_trap | sacadrl | ok | 20 | collisions | 0.8000 | 0.7000 | 0.8833 |
| cross_trap | sacadrl | ok | 20 | near_misses | 12.3000 | 8.6329 | 16.0667 |
| cross_trap | sacadrl | ok | 20 | snqi | -0.4434 | -0.5152 | -0.3616 |
| cross_trap | sacadrl | ok | 20 | success | 0.2000 | 0.1167 | 0.3000 |
| cross_trap | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.4167 | 0.2667 | 0.5667 |
| cross_trap | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 15.9667 | 9.0325 | 24.0500 |
| cross_trap | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.3094 | -0.4217 | -0.2088 |
| cross_trap | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.5000 | 0.3500 | 0.6500 |
| cross_trap | social_force | ok | 20 | collisions | 0.3333 | 0.2000 | 0.4833 |
| cross_trap | social_force | ok | 20 | near_misses | 17.9667 | 13.8992 | 22.1667 |
| cross_trap | social_force | ok | 20 | snqi | -1.0141 | -1.1015 | -0.9166 |
| cross_trap | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| cross_trap | socnav_sampling | ok | 20 | collisions | 0.9000 | 0.8167 | 0.9667 |
| cross_trap | socnav_sampling | ok | 20 | near_misses | 2.7833 | 1.7000 | 3.9833 |
| cross_trap | socnav_sampling | ok | 20 | snqi | -0.2937 | -0.3429 | -0.2458 |
| cross_trap | socnav_sampling | ok | 20 | success | 0.1000 | 0.0333 | 0.1833 |
| crossing | goal | ok | 20 | collisions | 0.3000 | 0.1000 | 0.5000 |
| crossing | goal | ok | 20 | near_misses | 9.1500 | 5.1000 | 13.6512 |
| crossing | goal | ok | 20 | snqi | -0.2683 | -0.3287 | -0.2086 |
| crossing | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| crossing | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| crossing | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 2.7500 | 0.6000 | 5.5000 |
| crossing | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.0053 | -0.1532 | 0.0875 |
| crossing | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| crossing | orca | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| crossing | orca | ok | 20 | near_misses | 2.7500 | 0.0000 | 7.3512 |
| crossing | orca | ok | 20 | snqi | 0.0490 | -0.0306 | 0.1060 |
| crossing | orca | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| crossing | ppo | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| crossing | ppo | ok | 20 | near_misses | 6.2000 | 2.7000 | 10.7500 |
| crossing | ppo | ok | 20 | snqi | -0.0924 | -0.1816 | -0.0177 |
| crossing | ppo | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| crossing | prediction_planner | ok | 20 | collisions | 0.3500 | 0.1500 | 0.5500 |
| crossing | prediction_planner | ok | 20 | near_misses | 7.2000 | 3.0488 | 12.0500 |
| crossing | prediction_planner | ok | 20 | snqi | -0.1229 | -0.2124 | -0.0383 |
| crossing | prediction_planner | ok | 20 | success | 0.6000 | 0.3988 | 0.8000 |
| crossing | sacadrl | ok | 20 | collisions | 0.9000 | 0.7500 | 1.0000 |
| crossing | sacadrl | ok | 20 | near_misses | 1.3000 | 0.0000 | 3.3025 |
| crossing | sacadrl | ok | 20 | snqi | -0.7257 | -0.7731 | -0.6684 |
| crossing | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| crossing | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| crossing | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 2.7500 | 0.6000 | 5.5000 |
| crossing | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.0053 | -0.1532 | 0.0875 |
| crossing | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| crossing | social_force | ok | 20 | collisions | 0.7000 | 0.5000 | 0.9000 |
| crossing | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| crossing | social_force | ok | 20 | snqi | -0.8649 | -0.8931 | -0.8392 |
| crossing | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| crossing | socnav_sampling | ok | 20 | collisions | 0.5000 | 0.2988 | 0.7000 |
| crossing | socnav_sampling | ok | 20 | near_misses | 0.6000 | 0.0000 | 1.4000 |
| crossing | socnav_sampling | ok | 20 | snqi | -0.0657 | -0.1433 | 0.0123 |
| crossing | socnav_sampling | ok | 20 | success | 0.5000 | 0.3000 | 0.7012 |
| crowd_navigation | goal | ok | 20 | collisions | 0.9000 | 0.7500 | 1.0000 |
| crowd_navigation | goal | ok | 20 | near_misses | 16.1500 | 12.2000 | 20.2500 |
| crowd_navigation | goal | ok | 20 | snqi | -0.4543 | -0.5036 | -0.4030 |
| crowd_navigation | goal | ok | 20 | success | 0.1000 | 0.0000 | 0.2500 |
| crowd_navigation | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.1000 | 0.0000 | 0.2500 |
| crowd_navigation | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 54.5500 | 30.3500 | 80.9175 |
| crowd_navigation | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.2475 | -0.3080 | -0.1880 |
| crowd_navigation | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| crowd_navigation | orca | ok | 20 | collisions | 0.1000 | 0.0000 | 0.2500 |
| crowd_navigation | orca | ok | 20 | near_misses | 10.4000 | 6.4500 | 14.6500 |
| crowd_navigation | orca | ok | 20 | snqi | -0.1957 | -0.2960 | -0.1101 |
| crowd_navigation | orca | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| crowd_navigation | ppo | ok | 20 | collisions | 0.1000 | 0.0000 | 0.2500 |
| crowd_navigation | ppo | ok | 20 | near_misses | 7.7500 | 4.8500 | 11.2512 |
| crowd_navigation | ppo | ok | 20 | snqi | -0.1936 | -0.2773 | -0.1290 |
| crowd_navigation | ppo | ok | 20 | success | 0.8500 | 0.6988 | 1.0000 |
| crowd_navigation | prediction_planner | ok | 20 | collisions | 0.6500 | 0.4500 | 0.8500 |
| crowd_navigation | prediction_planner | ok | 20 | near_misses | 38.2500 | 21.6000 | 56.2512 |
| crowd_navigation | prediction_planner | ok | 20 | snqi | -0.3634 | -0.4168 | -0.3137 |
| crowd_navigation | prediction_planner | ok | 20 | success | 0.3500 | 0.1500 | 0.5500 |
| crowd_navigation | sacadrl | ok | 20 | collisions | 0.1500 | 0.0000 | 0.3012 |
| crowd_navigation | sacadrl | ok | 20 | near_misses | 9.7500 | 5.2500 | 14.9512 |
| crowd_navigation | sacadrl | ok | 20 | snqi | -0.2199 | -0.3125 | -0.1375 |
| crowd_navigation | sacadrl | ok | 20 | success | 0.8500 | 0.6988 | 1.0000 |
| crowd_navigation | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.1000 | 0.0000 | 0.2500 |
| crowd_navigation | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 54.5500 | 30.3500 | 80.9175 |
| crowd_navigation | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.2475 | -0.3080 | -0.1880 |
| crowd_navigation | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| crowd_navigation | social_force | ok | 20 | collisions | 0.9500 | 0.8500 | 1.0000 |
| crowd_navigation | social_force | ok | 20 | near_misses | 16.0000 | 10.9000 | 22.5512 |
| crowd_navigation | social_force | ok | 20 | snqi | -0.4776 | -0.5501 | -0.4169 |
| crowd_navigation | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| crowd_navigation | socnav_sampling | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| crowd_navigation | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| crowd_navigation | socnav_sampling | ok | 20 | snqi | -0.3630 | -0.3815 | -0.3449 |
| crowd_navigation | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| doorway | goal | ok | 20 | collisions | 0.7833 | 0.6000 | 0.9500 |
| doorway | goal | ok | 20 | near_misses | 7.9667 | 5.7158 | 10.4171 |
| doorway | goal | ok | 20 | snqi | -0.3802 | -0.4594 | -0.2991 |
| doorway | goal | ok | 20 | success | 0.2167 | 0.0500 | 0.4000 |
| doorway | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.8333 | 0.7333 | 0.9167 |
| doorway | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 4.8167 | 3.6667 | 6.1504 |
| doorway | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.4267 | -0.5020 | -0.3497 |
| doorway | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.1500 | 0.0667 | 0.2500 |
| doorway | orca | ok | 20 | collisions | 0.4000 | 0.2667 | 0.5333 |
| doorway | orca | ok | 20 | near_misses | 6.4500 | 4.8163 | 8.3000 |
| doorway | orca | ok | 20 | snqi | -0.8687 | -0.9739 | -0.7571 |
| doorway | orca | ok | 20 | success | 0.2167 | 0.1167 | 0.3333 |
| doorway | ppo | ok | 20 | collisions | 0.5500 | 0.4333 | 0.6667 |
| doorway | ppo | ok | 20 | near_misses | 13.5333 | 11.3667 | 15.6167 |
| doorway | ppo | ok | 20 | snqi | -0.6233 | -0.7086 | -0.5354 |
| doorway | ppo | ok | 20 | success | 0.4333 | 0.3000 | 0.5667 |
| doorway | prediction_planner | ok | 20 | collisions | 0.8000 | 0.6833 | 0.9000 |
| doorway | prediction_planner | ok | 20 | near_misses | 9.9667 | 6.8667 | 13.2842 |
| doorway | prediction_planner | ok | 20 | snqi | -0.3895 | -0.4627 | -0.3224 |
| doorway | prediction_planner | ok | 20 | success | 0.2000 | 0.1000 | 0.3167 |
| doorway | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| doorway | sacadrl | ok | 20 | near_misses | 4.1500 | 3.0167 | 5.4167 |
| doorway | sacadrl | ok | 20 | snqi | -0.5085 | -0.5954 | -0.4285 |
| doorway | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| doorway | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.8333 | 0.7333 | 0.9167 |
| doorway | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 4.8167 | 3.6667 | 6.1504 |
| doorway | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.4229 | -0.4994 | -0.3456 |
| doorway | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.1500 | 0.0667 | 0.2500 |
| doorway | social_force | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| doorway | social_force | ok | 20 | near_misses | 7.5667 | 5.0667 | 10.6333 |
| doorway | social_force | ok | 20 | snqi | -1.2079 | -1.2411 | -1.1785 |
| doorway | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| doorway | socnav_sampling | ok | 20 | collisions | 1.0167 | 1.0000 | 1.0500 |
| doorway | socnav_sampling | ok | 20 | near_misses | 1.1500 | 0.6167 | 1.7833 |
| doorway | socnav_sampling | ok | 20 | snqi | -0.3975 | -0.4431 | -0.3537 |
| doorway | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| down_path | goal | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| down_path | goal | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| down_path | goal | ok | 20 | snqi | -0.1044 | -0.1122 | -0.0975 |
| down_path | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| down_path | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| down_path | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| down_path | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | 0.1070 | 0.0989 | 0.1140 |
| down_path | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| down_path | orca | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| down_path | orca | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| down_path | orca | ok | 20 | snqi | 0.1026 | 0.0987 | 0.1067 |
| down_path | orca | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| down_path | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| down_path | ppo | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| down_path | ppo | ok | 20 | snqi | 0.0932 | 0.0853 | 0.1009 |
| down_path | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| down_path | prediction_planner | ok | 20 | collisions | 0.1000 | 0.0000 | 0.2500 |
| down_path | prediction_planner | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| down_path | prediction_planner | ok | 20 | snqi | -0.0907 | -0.1201 | -0.0579 |
| down_path | prediction_planner | ok | 20 | success | 0.1000 | 0.0000 | 0.2500 |
| down_path | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| down_path | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| down_path | sacadrl | ok | 20 | snqi | -0.5338 | -0.5624 | -0.5037 |
| down_path | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| down_path | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| down_path | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| down_path | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | 0.1070 | 0.0989 | 0.1140 |
| down_path | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| down_path | social_force | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| down_path | social_force | ok | 20 | near_misses | 1.5000 | 0.3500 | 2.9500 |
| down_path | social_force | ok | 20 | snqi | -0.8622 | -0.8812 | -0.8463 |
| down_path | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| down_path | socnav_sampling | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| down_path | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| down_path | socnav_sampling | ok | 20 | snqi | -0.2820 | -0.3171 | -0.2476 |
| down_path | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| entering_elevator | goal | ok | 20 | collisions | 0.3000 | 0.1000 | 0.5000 |
| entering_elevator | goal | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| entering_elevator | goal | ok | 20 | snqi | -0.0855 | -0.1342 | -0.0329 |
| entering_elevator | goal | ok | 20 | success | 0.2500 | 0.0988 | 0.4500 |
| entering_elevator | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| entering_elevator | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 0.0500 | 0.0000 | 0.1500 |
| entering_elevator | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | 0.1103 | 0.0959 | 0.1212 |
| entering_elevator | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| entering_elevator | orca | ok | 20 | collisions | 0.1000 | 0.0000 | 0.2500 |
| entering_elevator | orca | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| entering_elevator | orca | ok | 20 | snqi | -0.1298 | -0.2492 | -0.0222 |
| entering_elevator | orca | ok | 20 | success | 0.6000 | 0.4000 | 0.8000 |
| entering_elevator | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| entering_elevator | ppo | ok | 20 | near_misses | 0.0500 | 0.0000 | 0.1500 |
| entering_elevator | ppo | ok | 20 | snqi | 0.0154 | -0.0087 | 0.0386 |
| entering_elevator | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| entering_elevator | prediction_planner | ok | 20 | collisions | 0.3000 | 0.1000 | 0.5000 |
| entering_elevator | prediction_planner | ok | 20 | near_misses | 0.1000 | 0.0000 | 0.3000 |
| entering_elevator | prediction_planner | ok | 20 | snqi | 0.0265 | -0.0386 | 0.0909 |
| entering_elevator | prediction_planner | ok | 20 | success | 0.7000 | 0.5000 | 0.9000 |
| entering_elevator | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| entering_elevator | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| entering_elevator | sacadrl | ok | 20 | snqi | -0.3811 | -0.4343 | -0.3275 |
| entering_elevator | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| entering_elevator | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| entering_elevator | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 0.0500 | 0.0000 | 0.1500 |
| entering_elevator | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | 0.1103 | 0.0959 | 0.1212 |
| entering_elevator | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| entering_elevator | social_force | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| entering_elevator | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| entering_elevator | social_force | ok | 20 | snqi | -0.7088 | -0.7180 | -0.6994 |
| entering_elevator | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| entering_elevator | socnav_sampling | ok | 20 | collisions | 0.5000 | 0.3000 | 0.7000 |
| entering_elevator | socnav_sampling | ok | 20 | near_misses | 0.0500 | 0.0000 | 0.1500 |
| entering_elevator | socnav_sampling | ok | 20 | snqi | -0.0398 | -0.1149 | 0.0338 |
| entering_elevator | socnav_sampling | ok | 20 | success | 0.5000 | 0.3000 | 0.7000 |
| entering_room | goal | ok | 20 | collisions | 0.3000 | 0.1000 | 0.5000 |
| entering_room | goal | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| entering_room | goal | ok | 20 | snqi | -0.1304 | -0.1513 | -0.1108 |
| entering_room | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| entering_room | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| entering_room | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| entering_room | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | 0.0932 | 0.0521 | 0.1178 |
| entering_room | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.9500 | 0.8500 | 1.0000 |
| entering_room | orca | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| entering_room | orca | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| entering_room | orca | ok | 20 | snqi | 0.1088 | 0.1003 | 0.1167 |
| entering_room | orca | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| entering_room | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| entering_room | ppo | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| entering_room | ppo | ok | 20 | snqi | 0.0287 | 0.0092 | 0.0487 |
| entering_room | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| entering_room | prediction_planner | ok | 20 | collisions | 0.3000 | 0.1000 | 0.5000 |
| entering_room | prediction_planner | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| entering_room | prediction_planner | ok | 20 | snqi | 0.0195 | -0.0434 | 0.0820 |
| entering_room | prediction_planner | ok | 20 | success | 0.7000 | 0.5000 | 0.9000 |
| entering_room | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| entering_room | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| entering_room | sacadrl | ok | 20 | snqi | -0.3222 | -0.3624 | -0.2834 |
| entering_room | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| entering_room | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| entering_room | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| entering_room | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | 0.0932 | 0.0521 | 0.1178 |
| entering_room | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.9500 | 0.8500 | 1.0000 |
| entering_room | social_force | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| entering_room | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| entering_room | social_force | ok | 20 | snqi | -0.9726 | -1.0041 | -0.9396 |
| entering_room | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| entering_room | socnav_sampling | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| entering_room | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| entering_room | socnav_sampling | ok | 20 | snqi | 0.1259 | 0.1240 | 0.1275 |
| entering_room | socnav_sampling | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| exiting_elevator | goal | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| exiting_elevator | goal | ok | 20 | near_misses | 7.3000 | 5.1500 | 9.3500 |
| exiting_elevator | goal | ok | 20 | snqi | -0.3315 | -0.3618 | -0.3001 |
| exiting_elevator | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| exiting_elevator | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| exiting_elevator | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 16.0500 | 13.6500 | 18.5500 |
| exiting_elevator | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.1940 | -0.2149 | -0.1762 |
| exiting_elevator | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| exiting_elevator | orca | ok | 20 | collisions | 0.4000 | 0.2000 | 0.6000 |
| exiting_elevator | orca | ok | 20 | near_misses | 4.4000 | 2.5500 | 6.4000 |
| exiting_elevator | orca | ok | 20 | snqi | -0.5570 | -0.6958 | -0.4224 |
| exiting_elevator | orca | ok | 20 | success | 0.6000 | 0.4000 | 0.8000 |
| exiting_elevator | ppo | ok | 20 | collisions | 0.1000 | 0.0000 | 0.2500 |
| exiting_elevator | ppo | ok | 20 | near_misses | 4.0000 | 1.8988 | 6.4500 |
| exiting_elevator | ppo | ok | 20 | snqi | -0.4600 | -0.6503 | -0.2963 |
| exiting_elevator | ppo | ok | 20 | success | 0.6500 | 0.4500 | 0.8500 |
| exiting_elevator | prediction_planner | ok | 20 | collisions | 0.3000 | 0.1000 | 0.5000 |
| exiting_elevator | prediction_planner | ok | 20 | near_misses | 35.3000 | 21.2488 | 49.2500 |
| exiting_elevator | prediction_planner | ok | 20 | snqi | -0.1682 | -0.1786 | -0.1575 |
| exiting_elevator | prediction_planner | ok | 20 | success | 0.7000 | 0.5000 | 0.9000 |
| exiting_elevator | sacadrl | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| exiting_elevator | sacadrl | ok | 20 | near_misses | 11.9000 | 10.3500 | 13.0500 |
| exiting_elevator | sacadrl | ok | 20 | snqi | -0.2433 | -0.2826 | -0.2059 |
| exiting_elevator | sacadrl | ok | 20 | success | 0.9500 | 0.8500 | 1.0000 |
| exiting_elevator | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| exiting_elevator | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 16.0500 | 13.6500 | 18.5500 |
| exiting_elevator | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.1940 | -0.2149 | -0.1762 |
| exiting_elevator | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| exiting_elevator | social_force | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| exiting_elevator | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| exiting_elevator | social_force | ok | 20 | snqi | -0.6747 | -0.7590 | -0.5892 |
| exiting_elevator | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| exiting_elevator | socnav_sampling | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| exiting_elevator | socnav_sampling | ok | 20 | near_misses | 2.1000 | 1.5000 | 2.7000 |
| exiting_elevator | socnav_sampling | ok | 20 | snqi | -0.2440 | -0.2548 | -0.2335 |
| exiting_elevator | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | goal | ok | 20 | collisions | 0.3000 | 0.1000 | 0.5000 |
| exiting_room | goal | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | goal | ok | 20 | snqi | -0.1294 | -0.1523 | -0.1080 |
| exiting_room | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | 0.1051 | 0.0901 | 0.1171 |
| exiting_room | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| exiting_room | orca | ok | 20 | collisions | 0.1000 | 0.0000 | 0.2500 |
| exiting_room | orca | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | orca | ok | 20 | snqi | 0.0422 | -0.0747 | 0.1221 |
| exiting_room | orca | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| exiting_room | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | ppo | ok | 20 | near_misses | 0.3000 | 0.0000 | 0.7000 |
| exiting_room | ppo | ok | 20 | snqi | 0.0144 | 0.0025 | 0.0253 |
| exiting_room | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| exiting_room | prediction_planner | ok | 20 | collisions | 0.3000 | 0.1000 | 0.5000 |
| exiting_room | prediction_planner | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | prediction_planner | ok | 20 | snqi | 0.0189 | -0.0439 | 0.0813 |
| exiting_room | prediction_planner | ok | 20 | success | 0.7000 | 0.5000 | 0.9000 |
| exiting_room | sacadrl | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | sacadrl | ok | 20 | snqi | -0.1875 | -0.2176 | -0.1573 |
| exiting_room | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | 0.1051 | 0.0901 | 0.1171 |
| exiting_room | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| exiting_room | social_force | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | social_force | ok | 20 | snqi | -0.8727 | -0.8918 | -0.8532 |
| exiting_room | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | socnav_sampling | ok | 20 | collisions | 0.5500 | 0.3500 | 0.7500 |
| exiting_room | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| exiting_room | socnav_sampling | ok | 20 | snqi | -0.0609 | -0.1329 | 0.0103 |
| exiting_room | socnav_sampling | ok | 20 | success | 0.4500 | 0.2500 | 0.6500 |
| following_human | goal | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| following_human | goal | ok | 20 | near_misses | 9.5000 | 4.8500 | 14.5500 |
| following_human | goal | ok | 20 | snqi | -0.2227 | -0.2851 | -0.1628 |
| following_human | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| following_human | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| following_human | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| following_human | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | 0.1060 | 0.0979 | 0.1130 |
| following_human | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| following_human | orca | ok | 20 | collisions | 0.7000 | 0.5000 | 0.9000 |
| following_human | orca | ok | 20 | near_misses | 11.3500 | 9.4500 | 13.6500 |
| following_human | orca | ok | 20 | snqi | -0.9222 | -1.1740 | -0.6393 |
| following_human | orca | ok | 20 | success | 0.3000 | 0.1000 | 0.5000 |
| following_human | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| following_human | ppo | ok | 20 | near_misses | 3.3000 | 1.5000 | 5.2500 |
| following_human | ppo | ok | 20 | snqi | 0.0471 | 0.0089 | 0.0806 |
| following_human | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| following_human | prediction_planner | ok | 20 | collisions | 0.1000 | 0.0000 | 0.2500 |
| following_human | prediction_planner | ok | 20 | near_misses | 8.2000 | 5.3000 | 11.1000 |
| following_human | prediction_planner | ok | 20 | snqi | -0.2299 | -0.2700 | -0.1906 |
| following_human | prediction_planner | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| following_human | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| following_human | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| following_human | sacadrl | ok | 20 | snqi | -0.6804 | -0.7297 | -0.6285 |
| following_human | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| following_human | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| following_human | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| following_human | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | 0.1060 | 0.0979 | 0.1130 |
| following_human | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| following_human | social_force | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| following_human | social_force | ok | 20 | near_misses | 17.4500 | 11.2988 | 23.9512 |
| following_human | social_force | ok | 20 | snqi | -1.0643 | -1.1530 | -0.9776 |
| following_human | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| following_human | socnav_sampling | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| following_human | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| following_human | socnav_sampling | ok | 20 | snqi | -0.2984 | -0.3336 | -0.2637 |
| following_human | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| frontal_approach | goal | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| frontal_approach | goal | ok | 20 | near_misses | 5.0000 | 5.0000 | 5.0000 |
| frontal_approach | goal | ok | 20 | snqi | -0.2732 | -0.2732 | -0.2731 |
| frontal_approach | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| frontal_approach | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| frontal_approach | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 7.5000 | 5.7500 | 9.7012 |
| frontal_approach | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.0158 | -0.0452 | 0.0126 |
| frontal_approach | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| frontal_approach | orca | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| frontal_approach | orca | ok | 20 | near_misses | 8.9500 | 8.8000 | 9.1000 |
| frontal_approach | orca | ok | 20 | snqi | -0.0116 | -0.0168 | -0.0068 |
| frontal_approach | orca | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| frontal_approach | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| frontal_approach | ppo | ok | 20 | near_misses | 5.0000 | 4.7000 | 5.3000 |
| frontal_approach | ppo | ok | 20 | snqi | 0.0059 | -0.0113 | 0.0210 |
| frontal_approach | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| frontal_approach | prediction_planner | ok | 20 | collisions | 0.9000 | 0.7500 | 1.0000 |
| frontal_approach | prediction_planner | ok | 20 | near_misses | 5.9000 | 4.6500 | 7.6500 |
| frontal_approach | prediction_planner | ok | 20 | snqi | -0.2558 | -0.2717 | -0.2356 |
| frontal_approach | prediction_planner | ok | 20 | success | 0.1000 | 0.0000 | 0.2500 |
| frontal_approach | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| frontal_approach | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| frontal_approach | sacadrl | ok | 20 | snqi | -0.6114 | -0.6428 | -0.5792 |
| frontal_approach | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| frontal_approach | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| frontal_approach | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 7.5000 | 5.7500 | 9.7012 |
| frontal_approach | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.0157 | -0.0448 | 0.0129 |
| frontal_approach | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| frontal_approach | social_force | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| frontal_approach | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| frontal_approach | social_force | ok | 20 | snqi | -0.7871 | -0.8042 | -0.7695 |
| frontal_approach | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| frontal_approach | socnav_sampling | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| frontal_approach | socnav_sampling | ok | 20 | near_misses | 2.2000 | 2.0500 | 2.4000 |
| frontal_approach | socnav_sampling | ok | 20 | snqi | -0.2329 | -0.2359 | -0.2304 |
| frontal_approach | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| group_crossing | goal | ok | 20 | collisions | 0.1500 | 0.0333 | 0.3167 |
| group_crossing | goal | ok | 20 | near_misses | 7.0500 | 3.9833 | 10.3508 |
| group_crossing | goal | ok | 20 | snqi | -0.2351 | -0.2825 | -0.1908 |
| group_crossing | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| group_crossing | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| group_crossing | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 18.3000 | 9.6158 | 28.3346 |
| group_crossing | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.0991 | -0.1666 | -0.0404 |
| group_crossing | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.9000 | 0.8000 | 0.9833 |
| group_crossing | orca | ok | 20 | collisions | 0.1833 | 0.0829 | 0.3167 |
| group_crossing | orca | ok | 20 | near_misses | 5.5667 | 3.5333 | 7.7000 |
| group_crossing | orca | ok | 20 | snqi | -0.2018 | -0.3294 | -0.0889 |
| group_crossing | orca | ok | 20 | success | 0.7333 | 0.5829 | 0.8833 |
| group_crossing | ppo | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1000 |
| group_crossing | ppo | ok | 20 | near_misses | 5.5167 | 3.2000 | 8.1500 |
| group_crossing | ppo | ok | 20 | snqi | -0.1890 | -0.2549 | -0.1254 |
| group_crossing | ppo | ok | 20 | success | 0.8167 | 0.6833 | 0.9333 |
| group_crossing | prediction_planner | ok | 20 | collisions | 0.2500 | 0.1000 | 0.4000 |
| group_crossing | prediction_planner | ok | 20 | near_misses | 12.6667 | 7.9325 | 18.3667 |
| group_crossing | prediction_planner | ok | 20 | snqi | -0.1223 | -0.1909 | -0.0507 |
| group_crossing | prediction_planner | ok | 20 | success | 0.7333 | 0.5667 | 0.8833 |
| group_crossing | sacadrl | ok | 20 | collisions | 0.1000 | 0.0167 | 0.2167 |
| group_crossing | sacadrl | ok | 20 | near_misses | 4.9333 | 2.4663 | 7.8337 |
| group_crossing | sacadrl | ok | 20 | snqi | -0.3283 | -0.3820 | -0.2798 |
| group_crossing | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| group_crossing | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| group_crossing | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 18.3000 | 9.6158 | 28.3346 |
| group_crossing | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.0991 | -0.1667 | -0.0404 |
| group_crossing | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.9000 | 0.8000 | 0.9833 |
| group_crossing | social_force | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| group_crossing | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| group_crossing | social_force | ok | 20 | snqi | -1.1054 | -1.1109 | -1.1006 |
| group_crossing | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| group_crossing | socnav_sampling | ok | 20 | collisions | 0.7667 | 0.6333 | 0.8833 |
| group_crossing | socnav_sampling | ok | 20 | near_misses | 2.8667 | 1.7167 | 4.1667 |
| group_crossing | socnav_sampling | ok | 20 | snqi | -0.2338 | -0.2888 | -0.1777 |
| group_crossing | socnav_sampling | ok | 20 | success | 0.2333 | 0.1167 | 0.3667 |
| head_on_corridor | goal | ok | 20 | collisions | 0.6500 | 0.4750 | 0.8250 |
| head_on_corridor | goal | ok | 20 | near_misses | 6.1500 | 4.4250 | 8.2250 |
| head_on_corridor | goal | ok | 20 | snqi | -0.3195 | -0.3730 | -0.2739 |
| head_on_corridor | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| head_on_corridor | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.4250 | 0.2500 | 0.6000 |
| head_on_corridor | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 5.2750 | 3.0750 | 7.7256 |
| head_on_corridor | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.1424 | -0.2086 | -0.0754 |
| head_on_corridor | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.5750 | 0.4000 | 0.7500 |
| head_on_corridor | orca | ok | 20 | collisions | 0.4000 | 0.2250 | 0.6000 |
| head_on_corridor | orca | ok | 20 | near_misses | 5.5750 | 3.8750 | 7.2750 |
| head_on_corridor | orca | ok | 20 | snqi | -0.1418 | -0.2162 | -0.0682 |
| head_on_corridor | orca | ok | 20 | success | 0.6000 | 0.4000 | 0.7750 |
| head_on_corridor | ppo | ok | 20 | collisions | 0.1000 | 0.0000 | 0.2500 |
| head_on_corridor | ppo | ok | 20 | near_misses | 2.7000 | 1.1250 | 4.5500 |
| head_on_corridor | ppo | ok | 20 | snqi | -0.0446 | -0.0869 | -0.0107 |
| head_on_corridor | ppo | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| head_on_corridor | prediction_planner | ok | 20 | collisions | 0.4250 | 0.2500 | 0.6006 |
| head_on_corridor | prediction_planner | ok | 20 | near_misses | 7.1000 | 4.0500 | 10.5750 |
| head_on_corridor | prediction_planner | ok | 20 | snqi | -0.1752 | -0.2567 | -0.0956 |
| head_on_corridor | prediction_planner | ok | 20 | success | 0.5750 | 0.3994 | 0.7500 |
| head_on_corridor | sacadrl | ok | 20 | collisions | 0.4500 | 0.3000 | 0.6250 |
| head_on_corridor | sacadrl | ok | 20 | near_misses | 9.1750 | 6.7500 | 11.7256 |
| head_on_corridor | sacadrl | ok | 20 | snqi | -0.4875 | -0.5482 | -0.4311 |
| head_on_corridor | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| head_on_corridor | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.4250 | 0.2500 | 0.6000 |
| head_on_corridor | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 5.2750 | 3.0750 | 7.7256 |
| head_on_corridor | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.1424 | -0.2086 | -0.0754 |
| head_on_corridor | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.5750 | 0.4000 | 0.7500 |
| head_on_corridor | social_force | ok | 20 | collisions | 0.6500 | 0.4750 | 0.8250 |
| head_on_corridor | social_force | ok | 20 | near_misses | 10.1250 | 6.6238 | 14.1000 |
| head_on_corridor | social_force | ok | 20 | snqi | -0.2753 | -0.3329 | -0.2122 |
| head_on_corridor | social_force | ok | 20 | success | 0.1250 | 0.0250 | 0.2500 |
| head_on_corridor | socnav_sampling | ok | 20 | collisions | 0.4000 | 0.2250 | 0.5750 |
| head_on_corridor | socnav_sampling | ok | 20 | near_misses | 2.7000 | 1.4744 | 4.0000 |
| head_on_corridor | socnav_sampling | ok | 20 | snqi | -0.1038 | -0.1591 | -0.0476 |
| head_on_corridor | socnav_sampling | ok | 20 | success | 0.6000 | 0.4250 | 0.7750 |
| intersection_no_gesture | goal | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | goal | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | goal | ok | 20 | snqi | -0.0960 | -0.0977 | -0.0949 |
| intersection_no_gesture | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | 0.0976 | 0.0656 | 0.1197 |
| intersection_no_gesture | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| intersection_no_gesture | orca | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | orca | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | orca | ok | 20 | snqi | 0.1155 | 0.1074 | 0.1213 |
| intersection_no_gesture | orca | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| intersection_no_gesture | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | ppo | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | ppo | ok | 20 | snqi | 0.0435 | 0.0307 | 0.0573 |
| intersection_no_gesture | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| intersection_no_gesture | prediction_planner | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | prediction_planner | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | prediction_planner | ok | 20 | snqi | 0.1062 | 0.1053 | 0.1070 |
| intersection_no_gesture | prediction_planner | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| intersection_no_gesture | sacadrl | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | sacadrl | ok | 20 | snqi | -0.2014 | -0.2382 | -0.1647 |
| intersection_no_gesture | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | 0.0976 | 0.0656 | 0.1197 |
| intersection_no_gesture | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| intersection_no_gesture | social_force | ok | 20 | collisions | 0.5500 | 0.3500 | 0.7500 |
| intersection_no_gesture | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | social_force | ok | 20 | snqi | -0.8758 | -0.8985 | -0.8529 |
| intersection_no_gesture | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | socnav_sampling | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_no_gesture | socnav_sampling | ok | 20 | snqi | 0.1211 | 0.1203 | 0.1218 |
| intersection_no_gesture | socnav_sampling | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| intersection_proceed | goal | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | goal | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | goal | ok | 20 | snqi | -0.0960 | -0.0977 | -0.0949 |
| intersection_proceed | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | 0.0976 | 0.0656 | 0.1197 |
| intersection_proceed | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| intersection_proceed | orca | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | orca | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | orca | ok | 20 | snqi | 0.1155 | 0.1074 | 0.1213 |
| intersection_proceed | orca | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| intersection_proceed | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | ppo | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | ppo | ok | 20 | snqi | 0.0458 | 0.0321 | 0.0607 |
| intersection_proceed | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| intersection_proceed | prediction_planner | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | prediction_planner | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | prediction_planner | ok | 20 | snqi | 0.1062 | 0.1053 | 0.1070 |
| intersection_proceed | prediction_planner | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| intersection_proceed | sacadrl | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | sacadrl | ok | 20 | snqi | -0.1985 | -0.2342 | -0.1629 |
| intersection_proceed | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | 0.0976 | 0.0656 | 0.1197 |
| intersection_proceed | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| intersection_proceed | social_force | ok | 20 | collisions | 0.5500 | 0.3500 | 0.7500 |
| intersection_proceed | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | social_force | ok | 20 | snqi | -0.8758 | -0.8985 | -0.8528 |
| intersection_proceed | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | socnav_sampling | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_proceed | socnav_sampling | ok | 20 | snqi | 0.1211 | 0.1203 | 0.1218 |
| intersection_proceed | socnav_sampling | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| intersection_wait | goal | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | goal | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | goal | ok | 20 | snqi | -0.0960 | -0.0977 | -0.0949 |
| intersection_wait | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | 0.0976 | 0.0656 | 0.1197 |
| intersection_wait | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| intersection_wait | orca | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | orca | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | orca | ok | 20 | snqi | 0.1155 | 0.1074 | 0.1213 |
| intersection_wait | orca | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| intersection_wait | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | ppo | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | ppo | ok | 20 | snqi | 0.0458 | 0.0321 | 0.0607 |
| intersection_wait | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| intersection_wait | prediction_planner | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | prediction_planner | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | prediction_planner | ok | 20 | snqi | 0.1062 | 0.1053 | 0.1070 |
| intersection_wait | prediction_planner | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| intersection_wait | sacadrl | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | sacadrl | ok | 20 | snqi | -0.1985 | -0.2342 | -0.1629 |
| intersection_wait | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | 0.0976 | 0.0656 | 0.1197 |
| intersection_wait | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| intersection_wait | social_force | ok | 20 | collisions | 0.5500 | 0.3500 | 0.7500 |
| intersection_wait | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | social_force | ok | 20 | snqi | -0.8758 | -0.8985 | -0.8528 |
| intersection_wait | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | socnav_sampling | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| intersection_wait | socnav_sampling | ok | 20 | snqi | 0.1211 | 0.1203 | 0.1218 |
| intersection_wait | socnav_sampling | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| join_group | goal | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| join_group | goal | ok | 20 | near_misses | 5.2500 | 5.1000 | 5.4500 |
| join_group | goal | ok | 20 | snqi | -0.2896 | -0.3002 | -0.2813 |
| join_group | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| join_group | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.1500 | 0.0000 | 0.3500 |
| join_group | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 20.9500 | 18.1500 | 23.6012 |
| join_group | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.3226 | -0.3606 | -0.2889 |
| join_group | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.8500 | 0.6500 | 1.0000 |
| join_group | orca | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| join_group | orca | ok | 20 | near_misses | 4.4000 | 4.0500 | 4.7500 |
| join_group | orca | ok | 20 | snqi | -0.3847 | -0.4073 | -0.3621 |
| join_group | orca | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| join_group | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| join_group | ppo | ok | 20 | near_misses | 8.9000 | 7.3500 | 10.1500 |
| join_group | ppo | ok | 20 | snqi | -0.1468 | -0.1849 | -0.1033 |
| join_group | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| join_group | prediction_planner | ok | 20 | collisions | 0.9500 | 0.8500 | 1.0000 |
| join_group | prediction_planner | ok | 20 | near_misses | 8.0000 | 6.8500 | 9.7500 |
| join_group | prediction_planner | ok | 20 | snqi | -0.3097 | -0.3223 | -0.2959 |
| join_group | prediction_planner | ok | 20 | success | 0.0500 | 0.0000 | 0.1500 |
| join_group | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| join_group | sacadrl | ok | 20 | near_misses | 7.6000 | 7.0500 | 8.1500 |
| join_group | sacadrl | ok | 20 | snqi | -0.4890 | -0.5372 | -0.4395 |
| join_group | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| join_group | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.1500 | 0.0000 | 0.3500 |
| join_group | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 20.9500 | 18.1500 | 23.6012 |
| join_group | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.3226 | -0.3606 | -0.2889 |
| join_group | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.8500 | 0.6500 | 1.0000 |
| join_group | social_force | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| join_group | social_force | ok | 20 | near_misses | 5.3000 | 5.1000 | 5.5000 |
| join_group | social_force | ok | 20 | snqi | -0.2826 | -0.2857 | -0.2797 |
| join_group | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| join_group | socnav_sampling | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| join_group | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| join_group | socnav_sampling | ok | 20 | snqi | 0.0840 | 0.0709 | 0.0961 |
| join_group | socnav_sampling | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| leading_human | goal | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| leading_human | goal | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| leading_human | goal | ok | 20 | snqi | -0.1052 | -0.1130 | -0.0984 |
| leading_human | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| leading_human | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| leading_human | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| leading_human | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | 0.1060 | 0.0979 | 0.1130 |
| leading_human | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| leading_human | orca | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| leading_human | orca | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| leading_human | orca | ok | 20 | snqi | 0.1145 | 0.1127 | 0.1160 |
| leading_human | orca | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| leading_human | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| leading_human | ppo | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| leading_human | ppo | ok | 20 | snqi | 0.0967 | 0.0853 | 0.1070 |
| leading_human | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| leading_human | prediction_planner | ok | 20 | collisions | 0.1000 | 0.0000 | 0.2500 |
| leading_human | prediction_planner | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| leading_human | prediction_planner | ok | 20 | snqi | -0.0647 | -0.1061 | -0.0211 |
| leading_human | prediction_planner | ok | 20 | success | 0.2500 | 0.1000 | 0.4500 |
| leading_human | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| leading_human | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| leading_human | sacadrl | ok | 20 | snqi | -0.6261 | -0.6735 | -0.5741 |
| leading_human | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| leading_human | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| leading_human | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| leading_human | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | 0.1060 | 0.0979 | 0.1130 |
| leading_human | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| leading_human | social_force | ok | 20 | collisions | 0.9500 | 0.8500 | 1.0000 |
| leading_human | social_force | ok | 20 | near_misses | 6.1000 | 2.6000 | 12.8500 |
| leading_human | social_force | ok | 20 | snqi | -0.8550 | -0.8911 | -0.8319 |
| leading_human | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| leading_human | socnav_sampling | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| leading_human | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| leading_human | socnav_sampling | ok | 20 | snqi | -0.3045 | -0.3403 | -0.2692 |
| leading_human | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| leave_group | goal | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| leave_group | goal | ok | 20 | near_misses | 6.6000 | 6.3500 | 6.8500 |
| leave_group | goal | ok | 20 | snqi | -0.3228 | -0.3419 | -0.3074 |
| leave_group | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| leave_group | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.1500 | 0.0000 | 0.3000 |
| leave_group | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 59.4000 | 44.8500 | 75.2025 |
| leave_group | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.5170 | -0.6278 | -0.4159 |
| leave_group | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.7000 | 0.5000 | 0.9000 |
| leave_group | orca | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| leave_group | orca | ok | 20 | near_misses | 9.9000 | 9.0000 | 10.9512 |
| leave_group | orca | ok | 20 | snqi | -0.5327 | -0.5437 | -0.5223 |
| leave_group | orca | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| leave_group | ppo | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| leave_group | ppo | ok | 20 | near_misses | 11.9500 | 10.2000 | 14.1000 |
| leave_group | ppo | ok | 20 | snqi | -0.2379 | -0.2879 | -0.1955 |
| leave_group | ppo | ok | 20 | success | 0.9500 | 0.8500 | 1.0000 |
| leave_group | prediction_planner | ok | 20 | collisions | 0.9000 | 0.7500 | 1.0000 |
| leave_group | prediction_planner | ok | 20 | near_misses | 12.5500 | 7.3000 | 19.4500 |
| leave_group | prediction_planner | ok | 20 | snqi | -0.3145 | -0.3545 | -0.2808 |
| leave_group | prediction_planner | ok | 20 | success | 0.1000 | 0.0000 | 0.2500 |
| leave_group | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| leave_group | sacadrl | ok | 20 | near_misses | 13.2000 | 10.6000 | 15.9000 |
| leave_group | sacadrl | ok | 20 | snqi | -0.5844 | -0.6373 | -0.5296 |
| leave_group | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| leave_group | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.4000 | 0.2000 | 0.6000 |
| leave_group | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 27.8500 | 23.5000 | 32.5500 |
| leave_group | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.4411 | -0.4972 | -0.3868 |
| leave_group | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.6000 | 0.4000 | 0.8000 |
| leave_group | social_force | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| leave_group | social_force | ok | 20 | near_misses | 8.0500 | 6.1500 | 10.2012 |
| leave_group | social_force | ok | 20 | snqi | -0.3202 | -0.3535 | -0.2909 |
| leave_group | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| leave_group | socnav_sampling | ok | 20 | collisions | 0.3500 | 0.1500 | 0.5500 |
| leave_group | socnav_sampling | ok | 20 | near_misses | 8.7500 | 6.8000 | 10.5500 |
| leave_group | socnav_sampling | ok | 20 | snqi | -0.2391 | -0.2637 | -0.2171 |
| leave_group | socnav_sampling | ok | 20 | success | 0.6500 | 0.4500 | 0.8500 |
| merging | goal | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| merging | goal | ok | 20 | near_misses | 30.4250 | 16.4988 | 44.7506 |
| merging | goal | ok | 20 | snqi | -0.5685 | -0.6293 | -0.5049 |
| merging | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| merging | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.4000 | 0.2250 | 0.5500 |
| merging | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 55.5000 | 39.3238 | 71.9006 |
| merging | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.5242 | -0.6045 | -0.4478 |
| merging | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.3250 | 0.1750 | 0.4750 |
| merging | orca | ok | 20 | collisions | 0.6500 | 0.5000 | 0.8000 |
| merging | orca | ok | 20 | near_misses | 65.2500 | 47.9738 | 83.4000 |
| merging | orca | ok | 20 | snqi | -0.5830 | -0.6593 | -0.5005 |
| merging | orca | ok | 20 | success | 0.2750 | 0.1500 | 0.4250 |
| merging | ppo | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| merging | ppo | ok | 20 | near_misses | 2.2750 | 0.8494 | 3.9756 |
| merging | ppo | ok | 20 | snqi | -0.3721 | -0.4354 | -0.3288 |
| merging | ppo | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| merging | prediction_planner | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| merging | prediction_planner | ok | 20 | near_misses | 5.5000 | 2.8250 | 8.2256 |
| merging | prediction_planner | ok | 20 | snqi | -0.4071 | -0.4540 | -0.3599 |
| merging | prediction_planner | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| merging | sacadrl | ok | 20 | collisions | 0.9750 | 0.9250 | 1.0000 |
| merging | sacadrl | ok | 20 | near_misses | 17.2250 | 9.3250 | 26.2256 |
| merging | sacadrl | ok | 20 | snqi | -0.5903 | -0.6420 | -0.5364 |
| merging | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| merging | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.5250 | 0.3744 | 0.6750 |
| merging | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 60.9500 | 42.7888 | 79.6094 |
| merging | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.5316 | -0.6013 | -0.4558 |
| merging | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.2500 | 0.1250 | 0.3750 |
| merging | social_force | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| merging | social_force | ok | 20 | near_misses | 0.2750 | 0.0000 | 0.8250 |
| merging | social_force | ok | 20 | snqi | -1.1271 | -1.1428 | -1.1164 |
| merging | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| merging | socnav_sampling | ok | 20 | collisions | 0.9500 | 0.8750 | 1.0000 |
| merging | socnav_sampling | ok | 20 | near_misses | 2.9000 | 1.6744 | 4.3500 |
| merging | socnav_sampling | ok | 20 | snqi | -0.3513 | -0.3844 | -0.3196 |
| merging | socnav_sampling | ok | 20 | success | 0.0500 | 0.0000 | 0.1250 |
| narrow_doorway | goal | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| narrow_doorway | goal | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | goal | ok | 20 | snqi | -0.2046 | -0.2105 | -0.2000 |
| narrow_doorway | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| narrow_doorway | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.2340 | -0.2476 | -0.2210 |
| narrow_doorway | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | orca | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | orca | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | orca | ok | 20 | snqi | -1.0949 | -1.0949 | -1.0949 |
| narrow_doorway | orca | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | ppo | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| narrow_doorway | ppo | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | ppo | ok | 20 | snqi | -0.2448 | -0.2630 | -0.2285 |
| narrow_doorway | ppo | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | prediction_planner | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| narrow_doorway | prediction_planner | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | prediction_planner | ok | 20 | snqi | -0.1997 | -0.1997 | -0.1997 |
| narrow_doorway | prediction_planner | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| narrow_doorway | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | sacadrl | ok | 20 | snqi | -0.4206 | -0.4618 | -0.3791 |
| narrow_doorway | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| narrow_doorway | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.2347 | -0.2484 | -0.2216 |
| narrow_doorway | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | social_force | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | social_force | ok | 20 | snqi | -0.8455 | -0.8527 | -0.8384 |
| narrow_doorway | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | socnav_sampling | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| narrow_doorway | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| narrow_doorway | socnav_sampling | ok | 20 | snqi | -0.2016 | -0.2046 | -0.1997 |
| narrow_doorway | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| narrow_hallway | goal | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| narrow_hallway | goal | ok | 20 | near_misses | 4.9000 | 4.7500 | 5.0000 |
| narrow_hallway | goal | ok | 20 | snqi | -0.2777 | -0.2815 | -0.2744 |
| narrow_hallway | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| narrow_hallway | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| narrow_hallway | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 86.8000 | 68.9000 | 103.6513 |
| narrow_hallway | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.2194 | -0.2686 | -0.1810 |
| narrow_hallway | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.8500 | 0.6500 | 1.0000 |
| narrow_hallway | orca | ok | 20 | collisions | 0.5000 | 0.3000 | 0.7000 |
| narrow_hallway | orca | ok | 20 | near_misses | 25.1000 | 17.0988 | 32.2500 |
| narrow_hallway | orca | ok | 20 | snqi | -0.5846 | -0.7946 | -0.3976 |
| narrow_hallway | orca | ok | 20 | success | 0.5000 | 0.3000 | 0.7000 |
| narrow_hallway | ppo | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| narrow_hallway | ppo | ok | 20 | near_misses | 1.1000 | 0.4000 | 1.9500 |
| narrow_hallway | ppo | ok | 20 | snqi | -0.3164 | -0.3455 | -0.2878 |
| narrow_hallway | ppo | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| narrow_hallway | prediction_planner | ok | 20 | collisions | 0.9000 | 0.7500 | 1.0000 |
| narrow_hallway | prediction_planner | ok | 20 | near_misses | 6.2000 | 4.7000 | 8.2500 |
| narrow_hallway | prediction_planner | ok | 20 | snqi | -0.2611 | -0.2798 | -0.2372 |
| narrow_hallway | prediction_planner | ok | 20 | success | 0.1000 | 0.0000 | 0.2500 |
| narrow_hallway | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| narrow_hallway | sacadrl | ok | 20 | near_misses | 6.1000 | 5.2500 | 7.0000 |
| narrow_hallway | sacadrl | ok | 20 | snqi | -0.4998 | -0.5485 | -0.4513 |
| narrow_hallway | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| narrow_hallway | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| narrow_hallway | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 86.8000 | 68.9000 | 103.6513 |
| narrow_hallway | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.2194 | -0.2686 | -0.1810 |
| narrow_hallway | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.8500 | 0.6500 | 1.0000 |
| narrow_hallway | social_force | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| narrow_hallway | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| narrow_hallway | social_force | ok | 20 | snqi | -0.6906 | -0.7045 | -0.6700 |
| narrow_hallway | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| narrow_hallway | socnav_sampling | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| narrow_hallway | socnav_sampling | ok | 20 | near_misses | 2.2500 | 2.1000 | 2.4500 |
| narrow_hallway | socnav_sampling | ok | 20 | snqi | -0.2402 | -0.2434 | -0.2373 |
| narrow_hallway | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| overtaking | goal | ok | 20 | collisions | 0.4500 | 0.2500 | 0.6500 |
| overtaking | goal | ok | 20 | near_misses | 3.4750 | 0.8000 | 6.8006 |
| overtaking | goal | ok | 20 | snqi | -0.2955 | -0.3630 | -0.2341 |
| overtaking | goal | ok | 20 | success | 0.2750 | 0.1250 | 0.4500 |
| overtaking | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.1000 | 0.0250 | 0.2000 |
| overtaking | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 16.1250 | 10.8744 | 21.5500 |
| overtaking | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.2737 | -0.3643 | -0.1824 |
| overtaking | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.9000 | 0.8000 | 0.9750 |
| overtaking | orca | ok | 20 | collisions | 0.3750 | 0.2250 | 0.5500 |
| overtaking | orca | ok | 20 | near_misses | 12.3750 | 7.1500 | 18.9512 |
| overtaking | orca | ok | 20 | snqi | -0.4566 | -0.5899 | -0.3351 |
| overtaking | orca | ok | 20 | success | 0.6000 | 0.4250 | 0.7500 |
| overtaking | ppo | ok | 20 | collisions | 0.1250 | 0.0250 | 0.2500 |
| overtaking | ppo | ok | 20 | near_misses | 10.6750 | 6.0500 | 15.9006 |
| overtaking | ppo | ok | 20 | snqi | -0.2691 | -0.3576 | -0.1903 |
| overtaking | ppo | ok | 20 | success | 0.8750 | 0.7500 | 0.9750 |
| overtaking | prediction_planner | ok | 20 | collisions | 0.5000 | 0.3000 | 0.7000 |
| overtaking | prediction_planner | ok | 20 | near_misses | 11.4000 | 5.3244 | 19.0500 |
| overtaking | prediction_planner | ok | 20 | snqi | -0.2638 | -0.3391 | -0.1872 |
| overtaking | prediction_planner | ok | 20 | success | 0.5000 | 0.3000 | 0.7000 |
| overtaking | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| overtaking | sacadrl | ok | 20 | near_misses | 5.9250 | 2.6994 | 9.6769 |
| overtaking | sacadrl | ok | 20 | snqi | -0.5277 | -0.5944 | -0.4687 |
| overtaking | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| overtaking | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.1000 | 0.0250 | 0.2000 |
| overtaking | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 15.3750 | 10.7975 | 19.9000 |
| overtaking | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.2745 | -0.3692 | -0.1808 |
| overtaking | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.9000 | 0.8000 | 0.9750 |
| overtaking | social_force | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| overtaking | social_force | ok | 20 | near_misses | 2.2000 | 0.4750 | 4.3000 |
| overtaking | social_force | ok | 20 | snqi | -1.1423 | -1.1730 | -1.1158 |
| overtaking | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| overtaking | socnav_sampling | ok | 20 | collisions | 0.6000 | 0.4500 | 0.7500 |
| overtaking | socnav_sampling | ok | 20 | near_misses | 4.3750 | 2.6494 | 6.3000 |
| overtaking | socnav_sampling | ok | 20 | snqi | -0.3641 | -0.4699 | -0.2572 |
| overtaking | socnav_sampling | ok | 20 | success | 0.4000 | 0.2500 | 0.5500 |
| parallel_traffic | goal | ok | 20 | collisions | 0.1500 | 0.0000 | 0.3000 |
| parallel_traffic | goal | ok | 20 | near_misses | 25.7500 | 15.4988 | 36.3500 |
| parallel_traffic | goal | ok | 20 | snqi | -0.3772 | -0.4417 | -0.3096 |
| parallel_traffic | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| parallel_traffic | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| parallel_traffic | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 23.9000 | 12.0500 | 37.7000 |
| parallel_traffic | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.1698 | -0.2537 | -0.0893 |
| parallel_traffic | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.8500 | 0.7000 | 1.0000 |
| parallel_traffic | orca | ok | 20 | collisions | 0.1000 | 0.0000 | 0.2500 |
| parallel_traffic | orca | ok | 20 | near_misses | 26.2500 | 19.7000 | 33.0000 |
| parallel_traffic | orca | ok | 20 | snqi | -0.3303 | -0.4307 | -0.2564 |
| parallel_traffic | orca | ok | 20 | success | 0.8500 | 0.7000 | 1.0000 |
| parallel_traffic | ppo | ok | 20 | collisions | 0.2500 | 0.0988 | 0.4500 |
| parallel_traffic | ppo | ok | 20 | near_misses | 11.5500 | 6.3488 | 17.1000 |
| parallel_traffic | ppo | ok | 20 | snqi | -0.2482 | -0.3367 | -0.1600 |
| parallel_traffic | ppo | ok | 20 | success | 0.7500 | 0.5500 | 0.9012 |
| parallel_traffic | prediction_planner | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| parallel_traffic | prediction_planner | ok | 20 | near_misses | 38.7500 | 26.3488 | 51.0000 |
| parallel_traffic | prediction_planner | ok | 20 | snqi | -0.3928 | -0.4519 | -0.3190 |
| parallel_traffic | prediction_planner | ok | 20 | success | 0.0500 | 0.0000 | 0.1500 |
| parallel_traffic | sacadrl | ok | 20 | collisions | 0.5000 | 0.3000 | 0.7000 |
| parallel_traffic | sacadrl | ok | 20 | near_misses | 18.0000 | 9.9500 | 27.1500 |
| parallel_traffic | sacadrl | ok | 20 | snqi | -0.5806 | -0.6884 | -0.4818 |
| parallel_traffic | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| parallel_traffic | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| parallel_traffic | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 23.9000 | 12.0500 | 37.7000 |
| parallel_traffic | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.1698 | -0.2537 | -0.0893 |
| parallel_traffic | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.8500 | 0.7000 | 1.0000 |
| parallel_traffic | social_force | ok | 20 | collisions | 0.9000 | 0.7500 | 1.0000 |
| parallel_traffic | social_force | ok | 20 | near_misses | 10.3000 | 6.5500 | 14.2512 |
| parallel_traffic | social_force | ok | 20 | snqi | -0.9295 | -1.0178 | -0.8384 |
| parallel_traffic | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| parallel_traffic | socnav_sampling | ok | 20 | collisions | 0.9500 | 0.8500 | 1.0000 |
| parallel_traffic | socnav_sampling | ok | 20 | near_misses | 2.0000 | 0.6000 | 3.5512 |
| parallel_traffic | socnav_sampling | ok | 20 | snqi | -0.3756 | -0.4620 | -0.2946 |
| parallel_traffic | socnav_sampling | ok | 20 | success | 0.0500 | 0.0000 | 0.1500 |
| pedestrian_obstruction | goal | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| pedestrian_obstruction | goal | ok | 20 | near_misses | 3.3500 | 1.8500 | 4.9000 |
| pedestrian_obstruction | goal | ok | 20 | snqi | -0.1508 | -0.1813 | -0.1243 |
| pedestrian_obstruction | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_obstruction | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| pedestrian_obstruction | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 9.9500 | 9.0000 | 10.8500 |
| pedestrian_obstruction | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.0404 | -0.0759 | -0.0143 |
| pedestrian_obstruction | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.9500 | 0.8500 | 1.0000 |
| pedestrian_obstruction | orca | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_obstruction | orca | ok | 20 | near_misses | 10.4500 | 10.2500 | 10.7000 |
| pedestrian_obstruction | orca | ok | 20 | snqi | -0.0230 | -0.0260 | -0.0201 |
| pedestrian_obstruction | orca | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| pedestrian_obstruction | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_obstruction | ppo | ok | 20 | near_misses | 4.0000 | 2.9000 | 5.1000 |
| pedestrian_obstruction | ppo | ok | 20 | snqi | 0.0159 | -0.0103 | 0.0401 |
| pedestrian_obstruction | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| pedestrian_obstruction | prediction_planner | ok | 20 | collisions | 0.6500 | 0.4500 | 0.8500 |
| pedestrian_obstruction | prediction_planner | ok | 20 | near_misses | 26.0500 | 21.4000 | 30.6012 |
| pedestrian_obstruction | prediction_planner | ok | 20 | snqi | -0.4124 | -0.4765 | -0.3476 |
| pedestrian_obstruction | prediction_planner | ok | 20 | success | 0.3500 | 0.1500 | 0.5500 |
| pedestrian_obstruction | sacadrl | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| pedestrian_obstruction | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_obstruction | sacadrl | ok | 20 | snqi | -0.7146 | -0.7395 | -0.6887 |
| pedestrian_obstruction | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_obstruction | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| pedestrian_obstruction | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 9.9500 | 9.0000 | 10.8500 |
| pedestrian_obstruction | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.0404 | -0.0759 | -0.0143 |
| pedestrian_obstruction | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.9500 | 0.8500 | 1.0000 |
| pedestrian_obstruction | social_force | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_obstruction | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_obstruction | social_force | ok | 20 | snqi | -0.7877 | -0.8019 | -0.7722 |
| pedestrian_obstruction | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_obstruction | socnav_sampling | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| pedestrian_obstruction | socnav_sampling | ok | 20 | near_misses | 3.1000 | 3.0000 | 3.2500 |
| pedestrian_obstruction | socnav_sampling | ok | 20 | snqi | -0.2496 | -0.2519 | -0.2480 |
| pedestrian_obstruction | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | goal | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | goal | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | goal | ok | 20 | snqi | -0.1067 | -0.1071 | -0.1064 |
| pedestrian_overtaking | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | 0.1042 | 0.1033 | 0.1050 |
| pedestrian_overtaking | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| pedestrian_overtaking | orca | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | orca | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | orca | ok | 20 | snqi | 0.1052 | 0.1045 | 0.1058 |
| pedestrian_overtaking | orca | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| pedestrian_overtaking | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | ppo | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | ppo | ok | 20 | snqi | 0.0691 | 0.0609 | 0.0784 |
| pedestrian_overtaking | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| pedestrian_overtaking | prediction_planner | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | prediction_planner | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | prediction_planner | ok | 20 | snqi | -0.0201 | -0.0588 | 0.0187 |
| pedestrian_overtaking | prediction_planner | ok | 20 | success | 0.4500 | 0.2500 | 0.6500 |
| pedestrian_overtaking | sacadrl | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | sacadrl | ok | 20 | near_misses | 0.3000 | 0.0000 | 0.9000 |
| pedestrian_overtaking | sacadrl | ok | 20 | snqi | -0.2255 | -0.2628 | -0.1945 |
| pedestrian_overtaking | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | 0.1042 | 0.1033 | 0.1050 |
| pedestrian_overtaking | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| pedestrian_overtaking | social_force | ok | 20 | collisions | 0.5500 | 0.3500 | 0.7500 |
| pedestrian_overtaking | social_force | ok | 20 | near_misses | 7.2500 | 4.0000 | 11.0012 |
| pedestrian_overtaking | social_force | ok | 20 | snqi | -0.9068 | -0.9596 | -0.8588 |
| pedestrian_overtaking | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | socnav_sampling | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| pedestrian_overtaking | socnav_sampling | ok | 20 | snqi | 0.1053 | 0.1047 | 0.1059 |
| pedestrian_overtaking | socnav_sampling | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| perpendicular_traffic | goal | ok | 20 | collisions | 0.4500 | 0.2500 | 0.6500 |
| perpendicular_traffic | goal | ok | 20 | near_misses | 10.4500 | 6.2000 | 15.1012 |
| perpendicular_traffic | goal | ok | 20 | snqi | -0.2901 | -0.3439 | -0.2353 |
| perpendicular_traffic | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| perpendicular_traffic | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| perpendicular_traffic | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 6.6000 | 1.9500 | 12.4512 |
| perpendicular_traffic | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.0094 | -0.1058 | 0.0701 |
| perpendicular_traffic | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| perpendicular_traffic | orca | ok | 20 | collisions | 0.3500 | 0.1500 | 0.5500 |
| perpendicular_traffic | orca | ok | 20 | near_misses | 13.7500 | 5.4500 | 25.2037 |
| perpendicular_traffic | orca | ok | 20 | snqi | -0.3383 | -0.5248 | -0.1600 |
| perpendicular_traffic | orca | ok | 20 | success | 0.4500 | 0.2500 | 0.6500 |
| perpendicular_traffic | ppo | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| perpendicular_traffic | ppo | ok | 20 | near_misses | 4.7500 | 1.4000 | 9.3512 |
| perpendicular_traffic | ppo | ok | 20 | snqi | -0.0579 | -0.1188 | -0.0059 |
| perpendicular_traffic | ppo | ok | 20 | success | 0.9500 | 0.8500 | 1.0000 |
| perpendicular_traffic | prediction_planner | ok | 20 | collisions | 0.1000 | 0.0000 | 0.2500 |
| perpendicular_traffic | prediction_planner | ok | 20 | near_misses | 11.8500 | 6.6500 | 17.4012 |
| perpendicular_traffic | prediction_planner | ok | 20 | snqi | -0.0758 | -0.1554 | 0.0019 |
| perpendicular_traffic | prediction_planner | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| perpendicular_traffic | sacadrl | ok | 20 | collisions | 0.5000 | 0.3000 | 0.7000 |
| perpendicular_traffic | sacadrl | ok | 20 | near_misses | 8.6000 | 5.7500 | 11.3500 |
| perpendicular_traffic | sacadrl | ok | 20 | snqi | -0.4313 | -0.5200 | -0.3444 |
| perpendicular_traffic | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| perpendicular_traffic | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| perpendicular_traffic | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 6.6000 | 1.9500 | 12.4512 |
| perpendicular_traffic | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.0095 | -0.1058 | 0.0701 |
| perpendicular_traffic | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.9000 | 0.7500 | 1.0000 |
| perpendicular_traffic | social_force | ok | 20 | collisions | 0.6000 | 0.4000 | 0.8000 |
| perpendicular_traffic | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| perpendicular_traffic | social_force | ok | 20 | snqi | -0.9017 | -0.9242 | -0.8786 |
| perpendicular_traffic | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| perpendicular_traffic | socnav_sampling | ok | 20 | collisions | 0.7500 | 0.5500 | 0.9500 |
| perpendicular_traffic | socnav_sampling | ok | 20 | near_misses | 1.9500 | 0.7000 | 3.4500 |
| perpendicular_traffic | socnav_sampling | ok | 20 | snqi | -0.2124 | -0.2736 | -0.1423 |
| perpendicular_traffic | socnav_sampling | ok | 20 | success | 0.2500 | 0.0500 | 0.4500 |
| robot_crowding | goal | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| robot_crowding | goal | ok | 20 | near_misses | 11.6500 | 8.3000 | 15.4000 |
| robot_crowding | goal | ok | 20 | snqi | -0.4692 | -0.5071 | -0.4323 |
| robot_crowding | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| robot_crowding | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.2000 | 0.0500 | 0.4000 |
| robot_crowding | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 103.8500 | 75.5950 | 131.6012 |
| robot_crowding | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.6324 | -0.7325 | -0.5383 |
| robot_crowding | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.8000 | 0.6000 | 0.9500 |
| robot_crowding | orca | ok | 20 | collisions | 0.4500 | 0.2500 | 0.6500 |
| robot_crowding | orca | ok | 20 | near_misses | 16.5500 | 11.3000 | 22.6500 |
| robot_crowding | orca | ok | 20 | snqi | -0.5503 | -0.6043 | -0.4966 |
| robot_crowding | orca | ok | 20 | success | 0.5500 | 0.3500 | 0.7500 |
| robot_crowding | ppo | ok | 20 | collisions | 0.2500 | 0.1000 | 0.4500 |
| robot_crowding | ppo | ok | 20 | near_misses | 9.0500 | 5.3500 | 12.9500 |
| robot_crowding | ppo | ok | 20 | snqi | -0.4565 | -0.5367 | -0.3818 |
| robot_crowding | ppo | ok | 20 | success | 0.7500 | 0.5500 | 0.9000 |
| robot_crowding | prediction_planner | ok | 20 | collisions | 0.8500 | 0.7000 | 1.0000 |
| robot_crowding | prediction_planner | ok | 20 | near_misses | 24.5000 | 12.6988 | 38.3537 |
| robot_crowding | prediction_planner | ok | 20 | snqi | -0.4515 | -0.4959 | -0.4020 |
| robot_crowding | prediction_planner | ok | 20 | success | 0.1500 | 0.0000 | 0.3000 |
| robot_crowding | sacadrl | ok | 20 | collisions | 0.8500 | 0.7000 | 1.0000 |
| robot_crowding | sacadrl | ok | 20 | near_misses | 18.6500 | 12.8000 | 25.8000 |
| robot_crowding | sacadrl | ok | 20 | snqi | -0.6226 | -0.6824 | -0.5648 |
| robot_crowding | sacadrl | ok | 20 | success | 0.1500 | 0.0000 | 0.3000 |
| robot_crowding | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.2000 | 0.0500 | 0.4000 |
| robot_crowding | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 103.8500 | 75.5950 | 131.6012 |
| robot_crowding | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.6324 | -0.7325 | -0.5383 |
| robot_crowding | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.8000 | 0.6000 | 0.9500 |
| robot_crowding | social_force | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| robot_crowding | social_force | ok | 20 | near_misses | 12.7000 | 8.3500 | 18.8512 |
| robot_crowding | social_force | ok | 20 | snqi | -0.4987 | -0.5917 | -0.4354 |
| robot_crowding | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| robot_crowding | socnav_sampling | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| robot_crowding | socnav_sampling | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| robot_crowding | socnav_sampling | ok | 20 | snqi | -1.0006 | -1.0797 | -0.9170 |
| robot_crowding | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| robot_overtaking | goal | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| robot_overtaking | goal | ok | 20 | near_misses | 9.6500 | 9.0000 | 10.4000 |
| robot_overtaking | goal | ok | 20 | snqi | -0.3701 | -0.3828 | -0.3579 |
| robot_overtaking | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| robot_overtaking | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| robot_overtaking | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 5.8500 | 4.1500 | 7.6000 |
| robot_overtaking | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | 0.0329 | 0.0065 | 0.0580 |
| robot_overtaking | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| robot_overtaking | orca | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| robot_overtaking | orca | ok | 20 | near_misses | 2.8000 | 1.6000 | 4.1000 |
| robot_overtaking | orca | ok | 20 | snqi | 0.0407 | 0.0214 | 0.0579 |
| robot_overtaking | orca | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| robot_overtaking | ppo | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| robot_overtaking | ppo | ok | 20 | near_misses | 3.3500 | 1.7500 | 5.1000 |
| robot_overtaking | ppo | ok | 20 | snqi | -0.0445 | -0.0642 | -0.0278 |
| robot_overtaking | ppo | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| robot_overtaking | prediction_planner | ok | 20 | collisions | 0.5000 | 0.3000 | 0.7000 |
| robot_overtaking | prediction_planner | ok | 20 | near_misses | 30.0000 | 24.0000 | 37.5012 |
| robot_overtaking | prediction_planner | ok | 20 | snqi | -0.4187 | -0.4964 | -0.3371 |
| robot_overtaking | prediction_planner | ok | 20 | success | 0.3500 | 0.1500 | 0.5500 |
| robot_overtaking | sacadrl | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| robot_overtaking | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| robot_overtaking | sacadrl | ok | 20 | snqi | -0.2794 | -0.3285 | -0.2318 |
| robot_overtaking | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| robot_overtaking | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0000 | 0.0000 | 0.0000 |
| robot_overtaking | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 5.8500 | 4.1500 | 7.6000 |
| robot_overtaking | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | 0.0329 | 0.0065 | 0.0580 |
| robot_overtaking | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 1.0000 | 1.0000 | 1.0000 |
| robot_overtaking | social_force | ok | 20 | collisions | 0.1500 | 0.0000 | 0.3000 |
| robot_overtaking | social_force | ok | 20 | near_misses | 1.6000 | 0.4000 | 3.1000 |
| robot_overtaking | social_force | ok | 20 | snqi | -0.7751 | -0.8150 | -0.7291 |
| robot_overtaking | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| robot_overtaking | socnav_sampling | ok | 20 | collisions | 0.8000 | 0.6000 | 0.9500 |
| robot_overtaking | socnav_sampling | ok | 20 | near_misses | 6.3500 | 5.1000 | 7.7512 |
| robot_overtaking | socnav_sampling | ok | 20 | snqi | -0.2494 | -0.2874 | -0.2069 |
| robot_overtaking | socnav_sampling | ok | 20 | success | 0.2000 | 0.0500 | 0.4000 |
| station_platform | goal | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| station_platform | goal | ok | 20 | near_misses | 14.0000 | 8.7500 | 19.7500 |
| station_platform | goal | ok | 20 | snqi | -0.4594 | -0.5114 | -0.4034 |
| station_platform | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| station_platform | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.5000 | 0.3000 | 0.7500 |
| station_platform | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 89.0500 | 53.6487 | 129.1000 |
| station_platform | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.4875 | -0.5484 | -0.4256 |
| station_platform | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.2000 | 0.0500 | 0.4000 |
| station_platform | orca | ok | 20 | collisions | 0.7000 | 0.5000 | 0.9000 |
| station_platform | orca | ok | 20 | near_misses | 33.2000 | 18.4488 | 51.4000 |
| station_platform | orca | ok | 20 | snqi | -0.6478 | -0.8047 | -0.5124 |
| station_platform | orca | ok | 20 | success | 0.1500 | 0.0000 | 0.3000 |
| station_platform | ppo | ok | 20 | collisions | 0.7000 | 0.5000 | 0.9000 |
| station_platform | ppo | ok | 20 | near_misses | 8.4000 | 4.3000 | 13.0012 |
| station_platform | ppo | ok | 20 | snqi | -0.4814 | -0.5444 | -0.4179 |
| station_platform | ppo | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| station_platform | prediction_planner | ok | 20 | collisions | 0.9000 | 0.7500 | 1.0000 |
| station_platform | prediction_planner | ok | 20 | near_misses | 73.0000 | 43.3000 | 102.7013 |
| station_platform | prediction_planner | ok | 20 | snqi | -0.4836 | -0.5515 | -0.4136 |
| station_platform | prediction_planner | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| station_platform | sacadrl | ok | 20 | collisions | 0.9500 | 0.8500 | 1.0000 |
| station_platform | sacadrl | ok | 20 | near_misses | 19.9000 | 12.2500 | 27.8500 |
| station_platform | sacadrl | ok | 20 | snqi | -0.6062 | -0.6896 | -0.5205 |
| station_platform | sacadrl | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| station_platform | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.5000 | 0.3000 | 0.7500 |
| station_platform | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 89.0500 | 53.6487 | 129.1000 |
| station_platform | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.4875 | -0.5484 | -0.4256 |
| station_platform | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.2000 | 0.0500 | 0.4000 |
| station_platform | social_force | ok | 20 | collisions | 0.3500 | 0.1500 | 0.5500 |
| station_platform | social_force | ok | 20 | near_misses | 6.4000 | 2.9988 | 10.1500 |
| station_platform | social_force | ok | 20 | snqi | -1.0519 | -1.2111 | -0.8782 |
| station_platform | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| station_platform | socnav_sampling | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| station_platform | socnav_sampling | ok | 20 | near_misses | 0.4500 | 0.0000 | 1.0000 |
| station_platform | socnav_sampling | ok | 20 | snqi | -0.4218 | -0.5015 | -0.3515 |
| station_platform | socnav_sampling | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| t_intersection | goal | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| t_intersection | goal | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| t_intersection | goal | ok | 20 | snqi | -0.3736 | -0.3988 | -0.3481 |
| t_intersection | goal | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| t_intersection | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| t_intersection | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | near_misses | 20.7750 | 8.7494 | 35.3506 |
| t_intersection | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | snqi | -0.1348 | -0.2177 | -0.0553 |
| t_intersection | hybrid_rule_v3_fast_progress_static_escape | ok | 20 | success | 0.9500 | 0.8500 | 1.0000 |
| t_intersection | orca | ok | 20 | collisions | 0.1750 | 0.0500 | 0.3000 |
| t_intersection | orca | ok | 20 | near_misses | 8.5750 | 4.5244 | 13.1000 |
| t_intersection | orca | ok | 20 | snqi | -0.1676 | -0.2538 | -0.0856 |
| t_intersection | orca | ok | 20 | success | 0.8250 | 0.7000 | 0.9500 |
| t_intersection | ppo | ok | 20 | collisions | 0.5500 | 0.3750 | 0.7250 |
| t_intersection | ppo | ok | 20 | near_misses | 0.7000 | 0.0000 | 1.5000 |
| t_intersection | ppo | ok | 20 | snqi | -0.2535 | -0.3195 | -0.1902 |
| t_intersection | ppo | ok | 20 | success | 0.4500 | 0.2750 | 0.6250 |
| t_intersection | prediction_planner | ok | 20 | collisions | 0.4250 | 0.2250 | 0.6500 |
| t_intersection | prediction_planner | ok | 20 | near_misses | 3.7750 | 0.0000 | 8.5756 |
| t_intersection | prediction_planner | ok | 20 | snqi | -0.1542 | -0.2470 | -0.0663 |
| t_intersection | prediction_planner | ok | 20 | success | 0.5750 | 0.3500 | 0.7750 |
| t_intersection | sacadrl | ok | 20 | collisions | 0.9750 | 0.9250 | 1.0000 |
| t_intersection | sacadrl | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| t_intersection | sacadrl | ok | 20 | snqi | -0.4366 | -0.4772 | -0.3965 |
| t_intersection | sacadrl | ok | 20 | success | 0.0250 | 0.0000 | 0.0750 |
| t_intersection | scenario_adaptive_hybrid_orca_v1 | ok | 20 | collisions | 0.0500 | 0.0000 | 0.1500 |
| t_intersection | scenario_adaptive_hybrid_orca_v1 | ok | 20 | near_misses | 20.8500 | 8.7994 | 35.4006 |
| t_intersection | scenario_adaptive_hybrid_orca_v1 | ok | 20 | snqi | -0.1349 | -0.2177 | -0.0553 |
| t_intersection | scenario_adaptive_hybrid_orca_v1 | ok | 20 | success | 0.9500 | 0.8500 | 1.0000 |
| t_intersection | social_force | ok | 20 | collisions | 1.0000 | 1.0000 | 1.0000 |
| t_intersection | social_force | ok | 20 | near_misses | 0.0000 | 0.0000 | 0.0000 |
| t_intersection | social_force | ok | 20 | snqi | -0.8585 | -0.8763 | -0.8399 |
| t_intersection | social_force | ok | 20 | success | 0.0000 | 0.0000 | 0.0000 |
| t_intersection | socnav_sampling | ok | 20 | collisions | 0.5250 | 0.3250 | 0.7250 |
| t_intersection | socnav_sampling | ok | 20 | near_misses | 3.8500 | 2.2500 | 5.5000 |
| t_intersection | socnav_sampling | ok | 20 | snqi | -0.1951 | -0.2941 | -0.0984 |
| t_intersection | socnav_sampling | ok | 20 | success | 0.4750 | 0.2750 | 0.6750 |

## Per-scenario rank stability

| scenario | identifiable | resamples | tau_mean | tau_min | flip_rate | top1_stable |
| --- | --- | --- | --- | --- | --- | --- |
| accompanying_peer | True | 500 | 0.9604 | 0.8333 | 0.0198 | False |
| blind_corner | True | 500 | 0.9132 | 0.6667 | 0.0434 | True |
| bottleneck | True | 500 | 0.9570 | 0.8333 | 0.0215 | False |
| circular_crossing | True | 500 | 0.7671 | 0.4444 | 0.1164 | False |
| cross_trap | True | 500 | 0.7790 | 0.2222 | 0.1105 | False |
| crossing | True | 500 | 0.8544 | 0.5000 | 0.0728 | False |
| crowd_navigation | True | 500 | 0.7898 | 0.4444 | 0.1051 | False |
| doorway | True | 500 | 0.7983 | 0.4444 | 0.1008 | False |
| down_path | True | 500 | 0.9413 | 0.7222 | 0.0293 | False |
| entering_elevator | True | 500 | 0.9113 | 0.6667 | 0.0443 | False |
| entering_room | True | 500 | 0.9232 | 0.7778 | 0.0384 | True |
| exiting_elevator | True | 500 | 0.9204 | 0.7778 | 0.0398 | False |
| exiting_room | True | 500 | 0.8836 | 0.6667 | 0.0582 | False |
| following_human | True | 500 | 0.9341 | 0.7778 | 0.0329 | False |
| frontal_approach | True | 500 | 0.9157 | 0.6667 | 0.0422 | False |
| group_crossing | True | 500 | 0.7989 | 0.5000 | 0.1006 | False |
| head_on_corridor | True | 500 | 0.8238 | 0.4444 | 0.0881 | False |
| intersection_no_gesture | True | 500 | 0.9088 | 0.6667 | 0.0456 | False |
| intersection_proceed | True | 500 | 0.9117 | 0.6667 | 0.0442 | False |
| intersection_wait | True | 500 | 0.9143 | 0.6667 | 0.0428 | False |
| join_group | True | 500 | 0.9343 | 0.7778 | 0.0328 | True |
| leading_human | True | 500 | 0.9622 | 0.7778 | 0.0189 | False |
| leave_group | True | 500 | 0.8668 | 0.6111 | 0.0666 | False |
| merging | True | 500 | 0.8342 | 0.5000 | 0.0829 | False |
| narrow_doorway | True | 500 | 0.9427 | 0.7222 | 0.0287 | False |
| narrow_hallway | True | 500 | 0.9209 | 0.6667 | 0.0396 | False |
| overtaking | True | 500 | 0.7283 | 0.3333 | 0.1358 | False |
| parallel_traffic | True | 500 | 0.8544 | 0.5000 | 0.0728 | False |
| pedestrian_obstruction | True | 500 | 0.9569 | 0.8333 | 0.0216 | True |
| pedestrian_overtaking | True | 500 | 0.9430 | 0.7778 | 0.0285 | False |
| perpendicular_traffic | True | 500 | 0.8761 | 0.6111 | 0.0619 | False |
| robot_crowding | True | 500 | 0.8020 | 0.5000 | 0.0990 | False |
| robot_overtaking | True | 500 | 0.9233 | 0.7222 | 0.0383 | False |
| station_platform | True | 500 | 0.7140 | 0.2222 | 0.1430 | False |
| t_intersection | True | 500 | 0.8070 | 0.5000 | 0.0965 | False |

## Adjacent-rank claim downgrades

| scenario | higher-rank planner | lower-rank planner | metric | decision | rationale |
| --- | --- | --- | --- | --- | --- |
| accompanying_peer | orca | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| accompanying_peer | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| accompanying_peer | scenario_adaptive_hybrid_orca_v1 | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| accompanying_peer | ppo | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| accompanying_peer | prediction_planner | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| accompanying_peer | goal | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| accompanying_peer | socnav_sampling | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| accompanying_peer | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| blind_corner | ppo | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| blind_corner | prediction_planner | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| blind_corner | socnav_sampling | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| blind_corner | orca | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| blind_corner | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| blind_corner | scenario_adaptive_hybrid_orca_v1 | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| blind_corner | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| blind_corner | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| bottleneck | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| bottleneck | scenario_adaptive_hybrid_orca_v1 | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| bottleneck | prediction_planner | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| bottleneck | socnav_sampling | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| bottleneck | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| bottleneck | sacadrl | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| bottleneck | ppo | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| bottleneck | orca | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| circular_crossing | prediction_planner | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| circular_crossing | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| circular_crossing | scenario_adaptive_hybrid_orca_v1 | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| circular_crossing | orca | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| circular_crossing | socnav_sampling | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| circular_crossing | ppo | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| circular_crossing | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| circular_crossing | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| cross_trap | ppo | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| cross_trap | socnav_sampling | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| cross_trap | scenario_adaptive_hybrid_orca_v1 | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| cross_trap | hybrid_rule_v3_fast_progress_static_escape | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| cross_trap | orca | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| cross_trap | prediction_planner | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| cross_trap | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| cross_trap | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crossing | orca | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crossing | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crossing | scenario_adaptive_hybrid_orca_v1 | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crossing | socnav_sampling | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crossing | ppo | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crossing | prediction_planner | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crossing | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crossing | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crowd_navigation | ppo | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crowd_navigation | orca | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crowd_navigation | sacadrl | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crowd_navigation | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crowd_navigation | scenario_adaptive_hybrid_orca_v1 | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crowd_navigation | socnav_sampling | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crowd_navigation | prediction_planner | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| crowd_navigation | goal | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| doorway | goal | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| doorway | prediction_planner | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| doorway | socnav_sampling | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| doorway | scenario_adaptive_hybrid_orca_v1 | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| doorway | hybrid_rule_v3_fast_progress_static_escape | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| doorway | sacadrl | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| doorway | ppo | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| doorway | orca | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| down_path | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| down_path | scenario_adaptive_hybrid_orca_v1 | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| down_path | orca | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| down_path | ppo | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| down_path | prediction_planner | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| down_path | goal | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| down_path | socnav_sampling | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| down_path | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_elevator | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_elevator | scenario_adaptive_hybrid_orca_v1 | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_elevator | prediction_planner | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_elevator | ppo | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_elevator | socnav_sampling | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_elevator | goal | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_elevator | orca | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_elevator | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_room | socnav_sampling | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_room | orca | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_room | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_room | scenario_adaptive_hybrid_orca_v1 | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_room | ppo | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_room | prediction_planner | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_room | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| entering_room | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_elevator | prediction_planner | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_elevator | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_elevator | scenario_adaptive_hybrid_orca_v1 | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_elevator | sacadrl | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_elevator | socnav_sampling | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_elevator | goal | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_elevator | ppo | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_elevator | orca | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_room | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_room | scenario_adaptive_hybrid_orca_v1 | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_room | orca | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_room | prediction_planner | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_room | ppo | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_room | socnav_sampling | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_room | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| exiting_room | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| following_human | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| following_human | scenario_adaptive_hybrid_orca_v1 | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| following_human | ppo | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| following_human | goal | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| following_human | prediction_planner | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| following_human | socnav_sampling | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| following_human | sacadrl | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| following_human | orca | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| frontal_approach | ppo | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| frontal_approach | orca | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| frontal_approach | scenario_adaptive_hybrid_orca_v1 | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| frontal_approach | hybrid_rule_v3_fast_progress_static_escape | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| frontal_approach | socnav_sampling | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| frontal_approach | prediction_planner | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| frontal_approach | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| frontal_approach | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| group_crossing | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| group_crossing | scenario_adaptive_hybrid_orca_v1 | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| group_crossing | prediction_planner | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| group_crossing | ppo | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| group_crossing | orca | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| group_crossing | socnav_sampling | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| group_crossing | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| group_crossing | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| head_on_corridor | ppo | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| head_on_corridor | socnav_sampling | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| head_on_corridor | orca | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| head_on_corridor | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| head_on_corridor | scenario_adaptive_hybrid_orca_v1 | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| head_on_corridor | prediction_planner | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| head_on_corridor | social_force | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| head_on_corridor | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_no_gesture | socnav_sampling | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_no_gesture | orca | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_no_gesture | prediction_planner | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_no_gesture | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_no_gesture | scenario_adaptive_hybrid_orca_v1 | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_no_gesture | ppo | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_no_gesture | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_no_gesture | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_proceed | socnav_sampling | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_proceed | orca | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_proceed | prediction_planner | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_proceed | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_proceed | scenario_adaptive_hybrid_orca_v1 | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_proceed | ppo | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_proceed | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_proceed | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_wait | socnav_sampling | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_wait | orca | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_wait | prediction_planner | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_wait | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_wait | scenario_adaptive_hybrid_orca_v1 | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_wait | ppo | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_wait | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| intersection_wait | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| join_group | socnav_sampling | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| join_group | ppo | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| join_group | social_force | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| join_group | goal | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| join_group | prediction_planner | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| join_group | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| join_group | scenario_adaptive_hybrid_orca_v1 | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| join_group | orca | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leading_human | orca | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leading_human | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leading_human | scenario_adaptive_hybrid_orca_v1 | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leading_human | ppo | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leading_human | prediction_planner | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leading_human | goal | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leading_human | socnav_sampling | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leading_human | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leave_group | ppo | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leave_group | socnav_sampling | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leave_group | prediction_planner | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leave_group | social_force | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leave_group | goal | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leave_group | scenario_adaptive_hybrid_orca_v1 | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leave_group | hybrid_rule_v3_fast_progress_static_escape | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| leave_group | orca | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| merging | socnav_sampling | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| merging | ppo | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| merging | prediction_planner | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| merging | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| merging | scenario_adaptive_hybrid_orca_v1 | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| merging | goal | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| merging | orca | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| merging | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_doorway | prediction_planner | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_doorway | socnav_sampling | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_doorway | goal | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_doorway | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_doorway | scenario_adaptive_hybrid_orca_v1 | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_doorway | ppo | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_doorway | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_doorway | social_force | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_hallway | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_hallway | scenario_adaptive_hybrid_orca_v1 | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_hallway | socnav_sampling | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_hallway | prediction_planner | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_hallway | goal | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_hallway | ppo | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_hallway | sacadrl | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| narrow_hallway | orca | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| overtaking | prediction_planner | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| overtaking | ppo | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| overtaking | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| overtaking | scenario_adaptive_hybrid_orca_v1 | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| overtaking | goal | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| overtaking | socnav_sampling | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| overtaking | orca | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| overtaking | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| parallel_traffic | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| parallel_traffic | scenario_adaptive_hybrid_orca_v1 | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| parallel_traffic | ppo | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| parallel_traffic | orca | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| parallel_traffic | socnav_sampling | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| parallel_traffic | goal | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| parallel_traffic | prediction_planner | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| parallel_traffic | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_obstruction | ppo | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_obstruction | orca | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_obstruction | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_obstruction | scenario_adaptive_hybrid_orca_v1 | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_obstruction | goal | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_obstruction | socnav_sampling | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_obstruction | prediction_planner | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_obstruction | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_overtaking | socnav_sampling | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_overtaking | orca | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_overtaking | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_overtaking | scenario_adaptive_hybrid_orca_v1 | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_overtaking | ppo | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_overtaking | prediction_planner | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_overtaking | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| pedestrian_overtaking | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| perpendicular_traffic | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| perpendicular_traffic | scenario_adaptive_hybrid_orca_v1 | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| perpendicular_traffic | ppo | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| perpendicular_traffic | prediction_planner | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| perpendicular_traffic | socnav_sampling | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| perpendicular_traffic | goal | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| perpendicular_traffic | orca | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| perpendicular_traffic | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_crowding | prediction_planner | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_crowding | ppo | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_crowding | goal | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_crowding | social_force | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_crowding | orca | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_crowding | sacadrl | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_crowding | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_crowding | scenario_adaptive_hybrid_orca_v1 | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_overtaking | orca | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_overtaking | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_overtaking | scenario_adaptive_hybrid_orca_v1 | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_overtaking | ppo | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_overtaking | socnav_sampling | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_overtaking | sacadrl | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_overtaking | goal | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| robot_overtaking | prediction_planner | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| station_platform | socnav_sampling | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| station_platform | goal | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| station_platform | ppo | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| station_platform | prediction_planner | hybrid_rule_v3_fast_progress_static_escape | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| station_platform | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| station_platform | scenario_adaptive_hybrid_orca_v1 | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| station_platform | sacadrl | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| station_platform | orca | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| t_intersection | hybrid_rule_v3_fast_progress_static_escape | scenario_adaptive_hybrid_orca_v1 | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| t_intersection | scenario_adaptive_hybrid_orca_v1 | prediction_planner | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| t_intersection | prediction_planner | orca | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| t_intersection | orca | socnav_sampling | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| t_intersection | socnav_sampling | ppo | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| t_intersection | ppo | goal | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| t_intersection | goal | sacadrl | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
| t_intersection | sacadrl | social_force | snqi | blocked_invalid_metric | rank metric 'snqi' is contract-invalid: SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247 |
## Manuscript/S30 decision packet

- **Manuscript table status**: `blocked`
- **S30 decision status**: `needs_review`
- **Minimum seed count**: 20
- **Adjacent CI-overlap downgrades**: 0
- **Diagnostic-only adjacent claims**: 0
- **Manuscript blockers**: invalid_rank_metric_contract
- **S30 reasons**: rank_metric_contract_invalid, rank_resampling_instability_present
- **Packet boundary**: Decision packet is local preflight only; no manuscript or paper claim is promoted.
