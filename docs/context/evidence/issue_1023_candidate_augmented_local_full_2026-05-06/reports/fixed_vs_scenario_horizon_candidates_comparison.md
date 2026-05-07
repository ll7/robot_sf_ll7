# Camera-Ready Campaign Comparison

- Base campaign: `paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217`
- Candidate campaign: `issue1023_scenario_horizons_candidates_local_2026-05-06`
- Reproducibility verdict: `drift_detected`

## Planner Deltas

| planner | base_status | candidate_status | base_episodes | candidate_episodes | exact_match | metric | base | candidate | delta(candidate-base) |
|---|---|---|---:|---:|---|---|---:|---:|---:|
| goal | ok | ok | 144 | 144 | no | success_mean | 0.0139 | 0.0556 | 0.0417 |
| goal | ok | ok | 144 | 144 | no | unfinished_mean | 0.9861 | 0.9444 | -0.0417 |
| goal | ok | ok | 144 | 144 | no | collisions_mean | 0.2361 | 0.6181 | 0.3820 |
| goal | ok | ok | 144 | 144 | no | near_misses_mean | 2.9097 | 7.0556 | 4.1459 |
| goal | ok | ok | 144 | 144 | no | snqi_mean | -0.1656 | -0.1904 | -0.0248 |
| goal | ok | ok | 144 | 144 | no | time_to_goal_norm_mean | 0.9981 | 0.9943 | -0.0038 |
| goal | ok | ok | 144 | 144 | no | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| goal | ok | ok | 144 | 144 | no | comfort_exposure_mean | 0.0297 | 0.0254 | -0.0043 |
| goal | ok | ok | 144 | 144 | no | jerk_mean | 0.0092 | 0.0082 | -0.0010 |
| orca | ok | ok | 144 | 144 | no | success_mean | 0.1806 | 0.7569 | 0.5763 |
| orca | ok | ok | 144 | 144 | no | unfinished_mean | 0.8194 | 0.2431 | -0.5763 |
| orca | ok | ok | 144 | 144 | no | collisions_mean | 0.0347 | 0.1667 | 0.1320 |
| orca | ok | ok | 144 | 144 | no | near_misses_mean | 4.9097 | 13.6806 | 8.7709 |
| orca | ok | ok | 144 | 144 | no | snqi_mean | -0.2589 | -0.2513 | 0.0076 |
| orca | ok | ok | 144 | 144 | no | time_to_goal_norm_mean | 0.9650 | 0.6833 | -0.2817 |
| orca | ok | ok | 144 | 144 | no | path_efficiency_mean | 1.0000 | 0.9896 | -0.0104 |
| orca | ok | ok | 144 | 144 | no | comfort_exposure_mean | 0.0310 | 0.0270 | -0.0040 |
| orca | ok | ok | 144 | 144 | no | jerk_mean | 0.1718 | 0.1874 | 0.0156 |
| ppo | ok | ok | 144 | 144 | no | success_mean | 0.2500 | 0.8056 | 0.5556 |
| ppo | ok | ok | 144 | 144 | no | unfinished_mean | 0.7500 | 0.1944 | -0.5556 |
| ppo | ok | ok | 144 | 144 | no | collisions_mean | 0.0903 | 0.1667 | 0.0764 |
| ppo | ok | ok | 144 | 144 | no | near_misses_mean | 3.3542 | 5.6181 | 2.2639 |
| ppo | ok | ok | 144 | 144 | no | snqi_mean | -0.3060 | -0.2074 | 0.0986 |
| ppo | ok | ok | 144 | 144 | no | time_to_goal_norm_mean | 0.9306 | 0.6165 | -0.3141 |
| ppo | ok | ok | 144 | 144 | no | path_efficiency_mean | 0.9866 | 0.9516 | -0.0350 |
| ppo | ok | ok | 144 | 144 | no | comfort_exposure_mean | 0.0292 | 0.0246 | -0.0046 |
| ppo | ok | ok | 144 | 144 | no | jerk_mean | 0.4629 | 0.4881 | 0.0252 |
| prediction_planner | ok | ok | 144 | 144 | no | success_mean | 0.0694 | 0.4931 | 0.4237 |
| prediction_planner | ok | ok | 144 | 144 | no | unfinished_mean | 0.9306 | 0.5069 | -0.4237 |
| prediction_planner | ok | ok | 144 | 144 | no | collisions_mean | 0.2083 | 0.4514 | 0.2431 |
| prediction_planner | ok | ok | 144 | 144 | no | near_misses_mean | 8.3681 | 24.9375 | 16.5694 |
| prediction_planner | ok | ok | 144 | 144 | no | snqi_mean | -0.1945 | -0.1408 | 0.0537 |
| prediction_planner | ok | ok | 144 | 144 | no | time_to_goal_norm_mean | 0.9867 | 0.8667 | -0.1200 |
| prediction_planner | ok | ok | 144 | 144 | no | path_efficiency_mean | 1.0000 | 0.9992 | -0.0008 |
| prediction_planner | ok | ok | 144 | 144 | no | comfort_exposure_mean | 0.0337 | 0.0323 | -0.0014 |
| prediction_planner | ok | ok | 144 | 144 | no | jerk_mean | 0.0835 | 0.0940 | 0.0105 |
| sacadrl | ok | ok | 144 | 144 | no | success_mean | 0.0000 | 0.0833 | 0.0833 |
| sacadrl | ok | ok | 144 | 144 | no | unfinished_mean | 1.0000 | 0.9167 | -0.0833 |
| sacadrl | ok | ok | 144 | 144 | no | collisions_mean | 0.3889 | 0.6667 | 0.2778 |
| sacadrl | ok | ok | 144 | 144 | no | near_misses_mean | 3.5694 | 5.7778 | 2.2084 |
| sacadrl | ok | ok | 144 | 144 | no | snqi_mean | -0.2834 | -0.2726 | 0.0108 |
| sacadrl | ok | ok | 144 | 144 | no | time_to_goal_norm_mean | 1.0000 | 0.9774 | -0.0226 |
| sacadrl | ok | ok | 144 | 144 | no | path_efficiency_mean | 1.0000 | 0.9989 | -0.0011 |
| sacadrl | ok | ok | 144 | 144 | no | comfort_exposure_mean | 0.0302 | 0.0240 | -0.0062 |
| sacadrl | ok | ok | 144 | 144 | no | jerk_mean | 0.0763 | 0.0704 | -0.0059 |
| social_force | ok | ok | 144 | 144 | no | success_mean | 0.0000 | 0.0139 | 0.0139 |
| social_force | ok | ok | 144 | 144 | no | unfinished_mean | 1.0000 | 0.9861 | -0.0139 |
| social_force | ok | ok | 144 | 144 | no | collisions_mean | 0.2083 | 0.3819 | 0.1736 |
| social_force | ok | ok | 144 | 144 | no | near_misses_mean | 2.3264 | 5.2917 | 2.9653 |
| social_force | ok | ok | 144 | 144 | no | snqi_mean | -0.8535 | -0.9537 | -0.1002 |
| social_force | ok | ok | 144 | 144 | no | time_to_goal_norm_mean | 1.0000 | 0.9985 | -0.0015 |
| social_force | ok | ok | 144 | 144 | no | path_efficiency_mean | 0.9362 | 0.8084 | -0.1278 |
| social_force | ok | ok | 144 | 144 | no | comfort_exposure_mean | 0.0435 | 0.0370 | -0.0065 |
| social_force | ok | ok | 144 | 144 | no | jerk_mean | 0.4937 | 0.4677 | -0.0260 |
| socnav_sampling | ok | ok | 144 | 144 | no | success_mean | 0.1736 | 0.4028 | 0.2292 |
| socnav_sampling | ok | ok | 144 | 144 | no | unfinished_mean | 0.8264 | 0.5972 | -0.2292 |
| socnav_sampling | ok | ok | 144 | 144 | no | collisions_mean | 0.5278 | 0.5972 | 0.0694 |
| socnav_sampling | ok | ok | 144 | 144 | no | near_misses_mean | 1.4097 | 1.6458 | 0.2361 |
| socnav_sampling | ok | ok | 144 | 144 | no | snqi_mean | -0.1390 | -0.0848 | 0.0542 |
| socnav_sampling | ok | ok | 144 | 144 | no | time_to_goal_norm_mean | 0.9528 | 0.8193 | -0.1335 |
| socnav_sampling | ok | ok | 144 | 144 | no | path_efficiency_mean | 0.9918 | 0.9886 | -0.0032 |
| socnav_sampling | ok | ok | 144 | 144 | no | comfort_exposure_mean | 0.0363 | 0.0341 | -0.0022 |
| socnav_sampling | ok | ok | 144 | 144 | no | jerk_mean | 0.0724 | 0.0719 | -0.0005 |

## Coverage Gaps

- Missing in base campaign: `hybrid_rule_v3_fast_progress_static_escape, scenario_adaptive_hybrid_orca_v1`
- Missing in candidate campaign: `socnav_bench`

## Reproducibility

- Status: `drift_detected`
- Mismatched planners: `goal, orca, ppo, prediction_planner, sacadrl, social_force, socnav_sampling`

## Scenario Deltas

- Complete machine-readable deltas are in the JSON artifact; showing up to 40 rows sorted by absolute success delta.
- Rows missing in base: `96`

| planner_key | scenario_family | scenario_id | base_episodes | candidate_episodes | metric | base | candidate | delta(candidate-base) |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| orca | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | snqi_mean | -0.0967 | 0.1232 | 0.2199 |
| orca | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6961 | -0.3039 |
| orca | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | comfort_exposure_mean | 0.0100 | 0.0066 | -0.0034 |
| orca | blind_corner | francis2023_blind_corner | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | blind_corner | francis2023_blind_corner | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | blind_corner | francis2023_blind_corner | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | blind_corner | francis2023_blind_corner | 3 | 3 | near_misses_mean | 0.0000 | 9.0000 | 9.0000 |
| orca | blind_corner | francis2023_blind_corner | 3 | 3 | snqi_mean | -0.0987 | -0.0225 | 0.0762 |
| orca | blind_corner | francis2023_blind_corner | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6804 | -0.3196 |
| orca | blind_corner | francis2023_blind_corner | 3 | 3 | path_efficiency_mean | 1.0000 | 0.9548 | -0.0452 |
| orca | blind_corner | francis2023_blind_corner | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | bottleneck | classic_bottleneck_low | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | bottleneck | classic_bottleneck_low | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | bottleneck | classic_bottleneck_low | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | bottleneck | classic_bottleneck_low | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | bottleneck | classic_bottleneck_low | 3 | 3 | snqi_mean | -0.0949 | 0.1317 | 0.2266 |
| orca | bottleneck | classic_bottleneck_low | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6193 | -0.3807 |
| orca | bottleneck | classic_bottleneck_low | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | bottleneck | classic_bottleneck_low | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | cross_trap | classic_cross_trap_low | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | cross_trap | classic_cross_trap_low | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | cross_trap | classic_cross_trap_low | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | cross_trap | classic_cross_trap_low | 3 | 3 | near_misses_mean | 2.3333 | 10.6667 | 8.3334 |
| orca | cross_trap | classic_cross_trap_low | 3 | 3 | snqi_mean | -0.1495 | 0.0376 | 0.1871 |
| orca | cross_trap | classic_cross_trap_low | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.4228 | -0.5772 |
| orca | cross_trap | classic_cross_trap_low | 3 | 3 | path_efficiency_mean | 1.0000 | 0.9592 | -0.0408 |
| orca | cross_trap | classic_cross_trap_low | 3 | 3 | comfort_exposure_mean | 0.0200 | 0.0092 | -0.0108 |
| orca | crossing | classic_urban_crossing_medium | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | crossing | classic_urban_crossing_medium | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | crossing | classic_urban_crossing_medium | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | crossing | classic_urban_crossing_medium | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | crossing | classic_urban_crossing_medium | 3 | 3 | snqi_mean | -0.1035 | 0.1173 | 0.2208 |
| orca | crossing | classic_urban_crossing_medium | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6814 | -0.3186 |
| orca | crossing | classic_urban_crossing_medium | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | crossing | classic_urban_crossing_medium | 3 | 3 | comfort_exposure_mean | 0.0113 | 0.0078 | -0.0035 |
| orca | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | near_misses_mean | 10.0000 | 11.6667 | 1.6667 |
| orca | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | snqi_mean | -0.8878 | -0.6002 | 0.2876 |
| orca | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.4782 | -0.5218 |
| orca | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | path_efficiency_mean | 1.0000 | 0.9974 | -0.0026 |
| orca | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | comfort_exposure_mean | 0.0833 | 0.0561 | -0.0272 |
| orca | down_path | francis2023_down_path | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | down_path | francis2023_down_path | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | down_path | francis2023_down_path | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | down_path | francis2023_down_path | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | down_path | francis2023_down_path | 3 | 3 | snqi_mean | -0.0949 | 0.1244 | 0.2193 |
| orca | down_path | francis2023_down_path | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6961 | -0.3039 |
| orca | down_path | francis2023_down_path | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | down_path | francis2023_down_path | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 3 | near_misses_mean | 11.0000 | 11.0000 | 0.0000 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 3 | snqi_mean | -0.6476 | -0.3915 | 0.2561 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.3938 | -0.6062 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | following_human | francis2023_following_human | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | following_human | francis2023_following_human | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | following_human | francis2023_following_human | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | following_human | francis2023_following_human | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | following_human | francis2023_following_human | 3 | 3 | snqi_mean | -0.0967 | 0.1232 | 0.2199 |
| orca | following_human | francis2023_following_human | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6961 | -0.3039 |
| orca | following_human | francis2023_following_human | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | following_human | francis2023_following_human | 3 | 3 | comfort_exposure_mean | 0.0100 | 0.0066 | -0.0034 |
| orca | frontal_approach | francis2023_frontal_approach | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | frontal_approach | francis2023_frontal_approach | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | frontal_approach | francis2023_frontal_approach | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | frontal_approach | francis2023_frontal_approach | 3 | 3 | near_misses_mean | 0.0000 | 9.3333 | 9.3333 |
| orca | frontal_approach | francis2023_frontal_approach | 3 | 3 | snqi_mean | -0.0949 | -0.0089 | 0.0860 |
| orca | frontal_approach | francis2023_frontal_approach | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6344 | -0.3656 |
| orca | frontal_approach | francis2023_frontal_approach | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | frontal_approach | francis2023_frontal_approach | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | near_misses_mean | 2.6667 | 2.6667 | 0.0000 |
| orca | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | snqi_mean | -0.1374 | 0.0943 | 0.2317 |
| orca | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.5573 | -0.4427 |
| orca | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | comfort_exposure_mean | 0.0083 | 0.0060 | -0.0023 |
| orca | head_on_corridor | classic_head_on_corridor_medium | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | head_on_corridor | classic_head_on_corridor_medium | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | head_on_corridor | classic_head_on_corridor_medium | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | head_on_corridor | classic_head_on_corridor_medium | 3 | 3 | near_misses_mean | 6.6667 | 9.0000 | 2.3333 |
| orca | head_on_corridor | classic_head_on_corridor_medium | 3 | 3 | snqi_mean | -0.2198 | -0.0250 | 0.1948 |
| orca | head_on_corridor | classic_head_on_corridor_medium | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.5195 | -0.4805 |
| orca | head_on_corridor | classic_head_on_corridor_medium | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | head_on_corridor | classic_head_on_corridor_medium | 3 | 3 | comfort_exposure_mean | 0.0358 | 0.0257 | -0.0101 |
| orca | leading_human | francis2023_leading_human | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | leading_human | francis2023_leading_human | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | leading_human | francis2023_leading_human | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | leading_human | francis2023_leading_human | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | leading_human | francis2023_leading_human | 3 | 3 | snqi_mean | -0.0967 | 0.1232 | 0.2199 |
| orca | leading_human | francis2023_leading_human | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6961 | -0.3039 |
| orca | leading_human | francis2023_leading_human | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | leading_human | francis2023_leading_human | 3 | 3 | comfort_exposure_mean | 0.0100 | 0.0066 | -0.0034 |
| orca | overtaking | classic_overtaking_low | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | overtaking | classic_overtaking_low | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | overtaking | classic_overtaking_low | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | overtaking | classic_overtaking_low | 3 | 3 | near_misses_mean | 0.0000 | 4.3333 | 4.3333 |
| orca | overtaking | classic_overtaking_low | 3 | 3 | snqi_mean | -0.2077 | -0.0454 | 0.1623 |
| orca | overtaking | classic_overtaking_low | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.5320 | -0.4680 |
| orca | overtaking | classic_overtaking_low | 3 | 3 | path_efficiency_mean | 1.0000 | 0.9637 | -0.0363 |
| orca | overtaking | classic_overtaking_low | 3 | 3 | comfort_exposure_mean | 0.0500 | 0.0287 | -0.0213 |
| orca | overtaking | classic_overtaking_medium | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | overtaking | classic_overtaking_medium | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | overtaking | classic_overtaking_medium | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | overtaking | classic_overtaking_medium | 3 | 3 | near_misses_mean | 6.6667 | 11.6667 | 5.0000 |
| orca | overtaking | classic_overtaking_medium | 3 | 3 | snqi_mean | -0.3894 | -0.1914 | 0.1980 |
| orca | overtaking | classic_overtaking_medium | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.5579 | -0.4421 |
| orca | overtaking | classic_overtaking_medium | 3 | 3 | path_efficiency_mean | 1.0000 | 0.9627 | -0.0373 |
| orca | overtaking | classic_overtaking_medium | 3 | 3 | comfort_exposure_mean | 0.0250 | 0.0115 | -0.0135 |
| orca | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | near_misses_mean | 6.6667 | 7.0000 | 0.3333 |
| orca | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | snqi_mean | -0.2708 | -0.0554 | 0.2154 |
| orca | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6427 | -0.3573 |
| orca | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | comfort_exposure_mean | 0.0696 | 0.0535 | -0.0161 |
| orca | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | near_misses_mean | 0.0000 | 12.3333 | 12.3333 |
| orca | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | snqi_mean | -0.0949 | -0.0522 | 0.0427 |
| orca | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.5827 | -0.4173 |
| orca | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0147 | 0.0147 |
| orca | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | snqi_mean | -0.1192 | 0.1028 | 0.2220 |
| orca | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.7072 | -0.2928 |
| orca | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | comfort_exposure_mean | 0.0800 | 0.0507 | -0.0293 |
| orca | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | near_misses_mean | 3.0000 | 3.0000 | 0.0000 |
| orca | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | snqi_mean | -0.1470 | 0.0905 | 0.2375 |
| orca | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.5048 | -0.4952 |
| orca | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | comfort_exposure_mean | 0.0081 | 0.0076 | -0.0005 |
| orca | robot_overtaking | francis2023_robot_overtaking | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | robot_overtaking | francis2023_robot_overtaking | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | robot_overtaking | francis2023_robot_overtaking | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | robot_overtaking | francis2023_robot_overtaking | 3 | 3 | near_misses_mean | 11.6667 | 11.6667 | 0.0000 |
| orca | robot_overtaking | francis2023_robot_overtaking | 3 | 3 | snqi_mean | -0.2662 | -0.0460 | 0.2202 |
| orca | robot_overtaking | francis2023_robot_overtaking | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6556 | -0.3444 |
| orca | robot_overtaking | francis2023_robot_overtaking | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | robot_overtaking | francis2023_robot_overtaking | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | t_intersection | classic_t_intersection_low | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | t_intersection | classic_t_intersection_low | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | t_intersection | classic_t_intersection_low | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | t_intersection | classic_t_intersection_low | 3 | 3 | near_misses_mean | 0.0000 | 15.6667 | 15.6667 |
| orca | t_intersection | classic_t_intersection_low | 3 | 3 | snqi_mean | -0.1684 | -0.0310 | 0.1374 |
| orca | t_intersection | classic_t_intersection_low | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.4511 | -0.5489 |
| orca | t_intersection | classic_t_intersection_low | 3 | 3 | path_efficiency_mean | 1.0000 | 0.9702 | -0.0298 |
| orca | t_intersection | classic_t_intersection_low | 3 | 3 | comfort_exposure_mean | 0.0283 | 0.0532 | 0.0249 |
| orca | t_intersection | classic_t_intersection_medium | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | t_intersection | classic_t_intersection_medium | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | t_intersection | classic_t_intersection_medium | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | t_intersection | classic_t_intersection_medium | 3 | 3 | near_misses_mean | 0.0000 | 16.6667 | 16.6667 |
| orca | t_intersection | classic_t_intersection_medium | 3 | 3 | snqi_mean | -0.1881 | -0.0571 | 0.1310 |
| orca | t_intersection | classic_t_intersection_medium | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.4558 | -0.5442 |
| orca | t_intersection | classic_t_intersection_medium | 3 | 3 | path_efficiency_mean | 1.0000 | 0.9843 | -0.0157 |
| orca | t_intersection | classic_t_intersection_medium | 3 | 3 | comfort_exposure_mean | 0.0567 | 0.0673 | 0.0106 |
| ppo | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | snqi_mean | -0.1003 | 0.1059 | 0.2062 |
| ppo | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6977 | -0.3023 |
| ppo | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | accompanying_peer | francis2023_accompanying_peer | 3 | 3 | comfort_exposure_mean | 0.0100 | 0.0066 | -0.0034 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 3 | snqi_mean | -0.2437 | -0.0264 | 0.2173 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6364 | -0.3636 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 3 | near_misses_mean | 5.0000 | 13.0000 | 8.0000 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 3 | snqi_mean | -0.3248 | -0.2134 | 0.1114 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.3384 | -0.6616 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | cross_trap | classic_cross_trap_low | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | cross_trap | classic_cross_trap_low | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | cross_trap | classic_cross_trap_low | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | cross_trap | classic_cross_trap_low | 3 | 3 | near_misses_mean | 1.0000 | 10.6667 | 9.6667 |
| ppo | cross_trap | classic_cross_trap_low | 3 | 3 | snqi_mean | -0.2848 | -0.1583 | 0.1265 |
| ppo | cross_trap | classic_cross_trap_low | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.4332 | -0.5668 |
| ppo | cross_trap | classic_cross_trap_low | 3 | 3 | path_efficiency_mean | 1.0000 | 0.9371 | -0.0629 |
| ppo | cross_trap | classic_cross_trap_low | 3 | 3 | comfort_exposure_mean | 0.0200 | 0.0089 | -0.0111 |
| ppo | cross_trap | classic_cross_trap_medium | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | cross_trap | classic_cross_trap_medium | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | cross_trap | classic_cross_trap_medium | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | cross_trap | classic_cross_trap_medium | 3 | 3 | near_misses_mean | 7.3333 | 25.0000 | 17.6667 |
| ppo | cross_trap | classic_cross_trap_medium | 3 | 3 | snqi_mean | -0.3715 | -0.3092 | 0.0623 |
| ppo | cross_trap | classic_cross_trap_medium | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.4314 | -0.5686 |
| ppo | cross_trap | classic_cross_trap_medium | 3 | 3 | path_efficiency_mean | 0.9702 | 0.6439 | -0.3263 |
| ppo | cross_trap | classic_cross_trap_medium | 3 | 3 | comfort_exposure_mean | 0.0121 | 0.0064 | -0.0057 |
| ppo | crossing | classic_urban_crossing_medium | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | crossing | classic_urban_crossing_medium | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | crossing | classic_urban_crossing_medium | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | crossing | classic_urban_crossing_medium | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | crossing | classic_urban_crossing_medium | 3 | 3 | snqi_mean | -0.2194 | -0.0002 | 0.2192 |
| ppo | crossing | classic_urban_crossing_medium | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.7139 | -0.2861 |
| ppo | crossing | classic_urban_crossing_medium | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | crossing | classic_urban_crossing_medium | 3 | 3 | comfort_exposure_mean | 0.0113 | 0.0074 | -0.0039 |
| ppo | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | near_misses_mean | 15.0000 | 15.3333 | 0.3333 |
| ppo | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | snqi_mean | -0.5435 | -0.4136 | 0.1299 |
| ppo | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.3671 | -0.6329 |
| ppo | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | crowd_navigation | francis2023_crowd_navigation | 3 | 3 | comfort_exposure_mean | 0.0830 | 0.0747 | -0.0083 |
| ppo | down_path | francis2023_down_path | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | down_path | francis2023_down_path | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | down_path | francis2023_down_path | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | down_path | francis2023_down_path | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | down_path | francis2023_down_path | 3 | 3 | snqi_mean | -0.1070 | 0.0968 | 0.2038 |
| ppo | down_path | francis2023_down_path | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6977 | -0.3023 |
| ppo | down_path | francis2023_down_path | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | down_path | francis2023_down_path | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | following_human | francis2023_following_human | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | following_human | francis2023_following_human | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | following_human | francis2023_following_human | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | following_human | francis2023_following_human | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | following_human | francis2023_following_human | 3 | 3 | snqi_mean | -0.1005 | 0.1136 | 0.2141 |
| ppo | following_human | francis2023_following_human | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6977 | -0.3023 |
| ppo | following_human | francis2023_following_human | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | following_human | francis2023_following_human | 3 | 3 | comfort_exposure_mean | 0.0100 | 0.0066 | -0.0034 |
| ppo | frontal_approach | francis2023_frontal_approach | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | frontal_approach | francis2023_frontal_approach | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | frontal_approach | francis2023_frontal_approach | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | frontal_approach | francis2023_frontal_approach | 3 | 3 | near_misses_mean | 0.0000 | 6.0000 | 6.0000 |
| ppo | frontal_approach | francis2023_frontal_approach | 3 | 3 | snqi_mean | -0.1027 | 0.0166 | 0.1193 |
| ppo | frontal_approach | francis2023_frontal_approach | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6277 | -0.3723 |
| ppo | frontal_approach | francis2023_frontal_approach | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | frontal_approach | francis2023_frontal_approach | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | near_misses_mean | 0.3333 | 0.6667 | 0.3334 |
| ppo | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | snqi_mean | -0.2132 | -0.3407 | -0.1275 |
| ppo | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.5776 | -0.4224 |
| ppo | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | head_on_corridor | classic_head_on_corridor_low | 3 | 3 | comfort_exposure_mean | 0.0083 | 0.0057 | -0.0026 |
| ppo | leading_human | francis2023_leading_human | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | leading_human | francis2023_leading_human | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | leading_human | francis2023_leading_human | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | leading_human | francis2023_leading_human | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | leading_human | francis2023_leading_human | 3 | 3 | snqi_mean | -0.1011 | 0.1083 | 0.2094 |
| ppo | leading_human | francis2023_leading_human | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6977 | -0.3023 |
| ppo | leading_human | francis2023_leading_human | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | leading_human | francis2023_leading_human | 3 | 3 | comfort_exposure_mean | 0.0100 | 0.0066 | -0.0034 |
| ppo | overtaking | classic_overtaking_low | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | overtaking | classic_overtaking_low | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | overtaking | classic_overtaking_low | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | overtaking | classic_overtaking_low | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | overtaking | classic_overtaking_low | 3 | 3 | snqi_mean | -0.3572 | -0.0936 | 0.2636 |
| ppo | overtaking | classic_overtaking_low | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.4828 | -0.5172 |
| ppo | overtaking | classic_overtaking_low | 3 | 3 | path_efficiency_mean | 1.0000 | 0.9836 | -0.0164 |
| ppo | overtaking | classic_overtaking_low | 3 | 3 | comfort_exposure_mean | 0.0500 | 0.0299 | -0.0201 |
| ppo | overtaking | classic_overtaking_medium | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | overtaking | classic_overtaking_medium | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | overtaking | classic_overtaking_medium | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | overtaking | classic_overtaking_medium | 3 | 3 | near_misses_mean | 0.0000 | 0.6667 | 0.6667 |
| ppo | overtaking | classic_overtaking_medium | 3 | 3 | snqi_mean | -0.3340 | -0.0820 | 0.2520 |
| ppo | overtaking | classic_overtaking_medium | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.4474 | -0.5526 |
| ppo | overtaking | classic_overtaking_medium | 3 | 3 | path_efficiency_mean | 1.0000 | 0.9790 | -0.0210 |
| ppo | overtaking | classic_overtaking_medium | 3 | 3 | comfort_exposure_mean | 0.0250 | 0.0150 | -0.0100 |
| ppo | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | near_misses_mean | 2.6667 | 3.0000 | 0.3333 |
| ppo | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | snqi_mean | -0.2849 | -0.0825 | 0.2024 |
| ppo | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6280 | -0.3720 |
| ppo | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | parallel_traffic | francis2023_parallel_traffic | 3 | 3 | comfort_exposure_mean | 0.0700 | 0.0539 | -0.0161 |
| ppo | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | near_misses_mean | 0.0000 | 5.3333 | 5.3333 |
| ppo | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | snqi_mean | -0.1077 | 0.0288 | 0.1365 |
| ppo | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.5778 | -0.4222 |
| ppo | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | pedestrian_obstruction | francis2023_pedestrian_obstruction | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | snqi_mean | -0.1235 | 0.0773 | 0.2008 |
| ppo | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.7042 | -0.2958 |
| ppo | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | pedestrian_overtaking | francis2023_pedestrian_overtaking | 3 | 3 | comfort_exposure_mean | 0.0800 | 0.0509 | -0.0291 |
| ppo | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | near_misses_mean | 5.3333 | 5.6667 | 0.3334 |
| ppo | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | snqi_mean | -0.4478 | -0.1945 | 0.2533 |
| ppo | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.5224 | -0.4776 |
| ppo | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | perpendicular_traffic | francis2023_perpendicular_traffic | 3 | 3 | comfort_exposure_mean | 0.0081 | 0.0073 | -0.0008 |

## Scenario Family Deltas

- Complete machine-readable deltas are in the JSON artifact; showing up to 40 rows sorted by absolute success delta.
- Rows missing in base: `70`

| planner_key | scenario_family | base_episodes | candidate_episodes | metric | base | candidate | delta(candidate-base) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| orca | accompanying_peer | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | accompanying_peer | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | accompanying_peer | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | accompanying_peer | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | accompanying_peer | 3 | 3 | snqi_mean | -0.0967 | 0.1232 | 0.2199 |
| orca | accompanying_peer | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6961 | -0.3039 |
| orca | accompanying_peer | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | accompanying_peer | 3 | 3 | comfort_exposure_mean | 0.0100 | 0.0066 | -0.0034 |
| orca | blind_corner | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | blind_corner | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | blind_corner | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | blind_corner | 3 | 3 | near_misses_mean | 0.0000 | 9.0000 | 9.0000 |
| orca | blind_corner | 3 | 3 | snqi_mean | -0.0987 | -0.0225 | 0.0762 |
| orca | blind_corner | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6804 | -0.3196 |
| orca | blind_corner | 3 | 3 | path_efficiency_mean | 1.0000 | 0.9548 | -0.0452 |
| orca | blind_corner | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | crossing | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | crossing | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | crossing | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | crossing | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | crossing | 3 | 3 | snqi_mean | -0.1035 | 0.1173 | 0.2208 |
| orca | crossing | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6814 | -0.3186 |
| orca | crossing | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | crossing | 3 | 3 | comfort_exposure_mean | 0.0113 | 0.0078 | -0.0035 |
| orca | crowd_navigation | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | crowd_navigation | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | crowd_navigation | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | crowd_navigation | 3 | 3 | near_misses_mean | 10.0000 | 11.6667 | 1.6667 |
| orca | crowd_navigation | 3 | 3 | snqi_mean | -0.8878 | -0.6002 | 0.2876 |
| orca | crowd_navigation | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.4782 | -0.5218 |
| orca | crowd_navigation | 3 | 3 | path_efficiency_mean | 1.0000 | 0.9974 | -0.0026 |
| orca | crowd_navigation | 3 | 3 | comfort_exposure_mean | 0.0833 | 0.0561 | -0.0272 |
| orca | down_path | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | down_path | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | down_path | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | down_path | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | down_path | 3 | 3 | snqi_mean | -0.0949 | 0.1244 | 0.2193 |
| orca | down_path | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6961 | -0.3039 |
| orca | down_path | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | down_path | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | exiting_elevator | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | exiting_elevator | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | exiting_elevator | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | exiting_elevator | 3 | 3 | near_misses_mean | 11.0000 | 11.0000 | 0.0000 |
| orca | exiting_elevator | 3 | 3 | snqi_mean | -0.6476 | -0.3915 | 0.2561 |
| orca | exiting_elevator | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.3938 | -0.6062 |
| orca | exiting_elevator | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | exiting_elevator | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | following_human | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | following_human | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | following_human | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | following_human | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | following_human | 3 | 3 | snqi_mean | -0.0967 | 0.1232 | 0.2199 |
| orca | following_human | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6961 | -0.3039 |
| orca | following_human | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | following_human | 3 | 3 | comfort_exposure_mean | 0.0100 | 0.0066 | -0.0034 |
| orca | frontal_approach | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | frontal_approach | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | frontal_approach | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | frontal_approach | 3 | 3 | near_misses_mean | 0.0000 | 9.3333 | 9.3333 |
| orca | frontal_approach | 3 | 3 | snqi_mean | -0.0949 | -0.0089 | 0.0860 |
| orca | frontal_approach | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6344 | -0.3656 |
| orca | frontal_approach | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | frontal_approach | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | head_on_corridor | 6 | 6 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | head_on_corridor | 6 | 6 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | head_on_corridor | 6 | 6 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | head_on_corridor | 6 | 6 | near_misses_mean | 4.6667 | 5.8333 | 1.1666 |
| orca | head_on_corridor | 6 | 6 | snqi_mean | -0.1786 | 0.0347 | 0.2133 |
| orca | head_on_corridor | 6 | 6 | time_to_goal_norm_mean | 1.0000 | 0.5384 | -0.4616 |
| orca | head_on_corridor | 6 | 6 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | head_on_corridor | 6 | 6 | comfort_exposure_mean | 0.0221 | 0.0158 | -0.0063 |
| orca | leading_human | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | leading_human | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | leading_human | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | leading_human | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | leading_human | 3 | 3 | snqi_mean | -0.0967 | 0.1232 | 0.2199 |
| orca | leading_human | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6961 | -0.3039 |
| orca | leading_human | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | leading_human | 3 | 3 | comfort_exposure_mean | 0.0100 | 0.0066 | -0.0034 |
| orca | overtaking | 6 | 6 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | overtaking | 6 | 6 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | overtaking | 6 | 6 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | overtaking | 6 | 6 | near_misses_mean | 3.3333 | 8.0000 | 4.6667 |
| orca | overtaking | 6 | 6 | snqi_mean | -0.2985 | -0.1184 | 0.1801 |
| orca | overtaking | 6 | 6 | time_to_goal_norm_mean | 1.0000 | 0.5450 | -0.4550 |
| orca | overtaking | 6 | 6 | path_efficiency_mean | 1.0000 | 0.9632 | -0.0368 |
| orca | overtaking | 6 | 6 | comfort_exposure_mean | 0.0375 | 0.0201 | -0.0174 |
| orca | parallel_traffic | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | parallel_traffic | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | parallel_traffic | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | parallel_traffic | 3 | 3 | near_misses_mean | 6.6667 | 7.0000 | 0.3333 |
| orca | parallel_traffic | 3 | 3 | snqi_mean | -0.2708 | -0.0554 | 0.2154 |
| orca | parallel_traffic | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6427 | -0.3573 |
| orca | parallel_traffic | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | parallel_traffic | 3 | 3 | comfort_exposure_mean | 0.0696 | 0.0535 | -0.0161 |
| orca | pedestrian_obstruction | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | pedestrian_obstruction | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | pedestrian_obstruction | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | pedestrian_obstruction | 3 | 3 | near_misses_mean | 0.0000 | 12.3333 | 12.3333 |
| orca | pedestrian_obstruction | 3 | 3 | snqi_mean | -0.0949 | -0.0522 | 0.0427 |
| orca | pedestrian_obstruction | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.5827 | -0.4173 |
| orca | pedestrian_obstruction | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | pedestrian_obstruction | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0147 | 0.0147 |
| orca | pedestrian_overtaking | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | pedestrian_overtaking | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | pedestrian_overtaking | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | pedestrian_overtaking | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | pedestrian_overtaking | 3 | 3 | snqi_mean | -0.1192 | 0.1028 | 0.2220 |
| orca | pedestrian_overtaking | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.7072 | -0.2928 |
| orca | pedestrian_overtaking | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | pedestrian_overtaking | 3 | 3 | comfort_exposure_mean | 0.0800 | 0.0507 | -0.0293 |
| orca | perpendicular_traffic | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | perpendicular_traffic | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | perpendicular_traffic | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | perpendicular_traffic | 3 | 3 | near_misses_mean | 3.0000 | 3.0000 | 0.0000 |
| orca | perpendicular_traffic | 3 | 3 | snqi_mean | -0.1470 | 0.0905 | 0.2375 |
| orca | perpendicular_traffic | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.5048 | -0.4952 |
| orca | perpendicular_traffic | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | perpendicular_traffic | 3 | 3 | comfort_exposure_mean | 0.0081 | 0.0076 | -0.0005 |
| orca | robot_overtaking | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | robot_overtaking | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | robot_overtaking | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | robot_overtaking | 3 | 3 | near_misses_mean | 11.6667 | 11.6667 | 0.0000 |
| orca | robot_overtaking | 3 | 3 | snqi_mean | -0.2662 | -0.0460 | 0.2202 |
| orca | robot_overtaking | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6556 | -0.3444 |
| orca | robot_overtaking | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | robot_overtaking | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | t_intersection | 6 | 6 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| orca | t_intersection | 6 | 6 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| orca | t_intersection | 6 | 6 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | t_intersection | 6 | 6 | near_misses_mean | 0.0000 | 16.1667 | 16.1667 |
| orca | t_intersection | 6 | 6 | snqi_mean | -0.1782 | -0.0440 | 0.1342 |
| orca | t_intersection | 6 | 6 | time_to_goal_norm_mean | 1.0000 | 0.4535 | -0.5465 |
| orca | t_intersection | 6 | 6 | path_efficiency_mean | 1.0000 | 0.9773 | -0.0227 |
| orca | t_intersection | 6 | 6 | comfort_exposure_mean | 0.0425 | 0.0603 | 0.0178 |
| ppo | accompanying_peer | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | accompanying_peer | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | accompanying_peer | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | accompanying_peer | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | accompanying_peer | 3 | 3 | snqi_mean | -0.1003 | 0.1059 | 0.2062 |
| ppo | accompanying_peer | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6977 | -0.3023 |
| ppo | accompanying_peer | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | accompanying_peer | 3 | 3 | comfort_exposure_mean | 0.0100 | 0.0066 | -0.0034 |
| ppo | crossing | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | crossing | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | crossing | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | crossing | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | crossing | 3 | 3 | snqi_mean | -0.2194 | -0.0002 | 0.2192 |
| ppo | crossing | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.7139 | -0.2861 |
| ppo | crossing | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | crossing | 3 | 3 | comfort_exposure_mean | 0.0113 | 0.0074 | -0.0039 |
| ppo | crowd_navigation | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | crowd_navigation | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | crowd_navigation | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | crowd_navigation | 3 | 3 | near_misses_mean | 15.0000 | 15.3333 | 0.3333 |
| ppo | crowd_navigation | 3 | 3 | snqi_mean | -0.5435 | -0.4136 | 0.1299 |
| ppo | crowd_navigation | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.3671 | -0.6329 |
| ppo | crowd_navigation | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | crowd_navigation | 3 | 3 | comfort_exposure_mean | 0.0830 | 0.0747 | -0.0083 |
| ppo | down_path | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | down_path | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | down_path | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | down_path | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | down_path | 3 | 3 | snqi_mean | -0.1070 | 0.0968 | 0.2038 |
| ppo | down_path | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6977 | -0.3023 |
| ppo | down_path | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | down_path | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | following_human | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | following_human | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | following_human | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | following_human | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | following_human | 3 | 3 | snqi_mean | -0.1005 | 0.1136 | 0.2141 |
| ppo | following_human | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6977 | -0.3023 |
| ppo | following_human | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | following_human | 3 | 3 | comfort_exposure_mean | 0.0100 | 0.0066 | -0.0034 |
| ppo | frontal_approach | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | frontal_approach | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | frontal_approach | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | frontal_approach | 3 | 3 | near_misses_mean | 0.0000 | 6.0000 | 6.0000 |
| ppo | frontal_approach | 3 | 3 | snqi_mean | -0.1027 | 0.0166 | 0.1193 |
| ppo | frontal_approach | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6277 | -0.3723 |
| ppo | frontal_approach | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | frontal_approach | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | leading_human | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | leading_human | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | leading_human | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | leading_human | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | leading_human | 3 | 3 | snqi_mean | -0.1011 | 0.1083 | 0.2094 |
| ppo | leading_human | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6977 | -0.3023 |
| ppo | leading_human | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | leading_human | 3 | 3 | comfort_exposure_mean | 0.0100 | 0.0066 | -0.0034 |
| ppo | overtaking | 6 | 6 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | overtaking | 6 | 6 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | overtaking | 6 | 6 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | overtaking | 6 | 6 | near_misses_mean | 0.0000 | 0.3333 | 0.3333 |
| ppo | overtaking | 6 | 6 | snqi_mean | -0.3456 | -0.0878 | 0.2578 |
| ppo | overtaking | 6 | 6 | time_to_goal_norm_mean | 1.0000 | 0.4651 | -0.5349 |
| ppo | overtaking | 6 | 6 | path_efficiency_mean | 1.0000 | 0.9813 | -0.0187 |
| ppo | overtaking | 6 | 6 | comfort_exposure_mean | 0.0375 | 0.0225 | -0.0150 |
| ppo | parallel_traffic | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | parallel_traffic | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | parallel_traffic | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | parallel_traffic | 3 | 3 | near_misses_mean | 2.6667 | 3.0000 | 0.3333 |
| ppo | parallel_traffic | 3 | 3 | snqi_mean | -0.2849 | -0.0825 | 0.2024 |
| ppo | parallel_traffic | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6280 | -0.3720 |
| ppo | parallel_traffic | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | parallel_traffic | 3 | 3 | comfort_exposure_mean | 0.0700 | 0.0539 | -0.0161 |
| ppo | pedestrian_obstruction | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | pedestrian_obstruction | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | pedestrian_obstruction | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | pedestrian_obstruction | 3 | 3 | near_misses_mean | 0.0000 | 5.3333 | 5.3333 |
| ppo | pedestrian_obstruction | 3 | 3 | snqi_mean | -0.1077 | 0.0288 | 0.1365 |
| ppo | pedestrian_obstruction | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.5778 | -0.4222 |
| ppo | pedestrian_obstruction | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | pedestrian_obstruction | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | pedestrian_overtaking | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | pedestrian_overtaking | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | pedestrian_overtaking | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | pedestrian_overtaking | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | pedestrian_overtaking | 3 | 3 | snqi_mean | -0.1235 | 0.0773 | 0.2008 |
| ppo | pedestrian_overtaking | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.7042 | -0.2958 |
| ppo | pedestrian_overtaking | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | pedestrian_overtaking | 3 | 3 | comfort_exposure_mean | 0.0800 | 0.0509 | -0.0291 |
| ppo | perpendicular_traffic | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | perpendicular_traffic | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | perpendicular_traffic | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | perpendicular_traffic | 3 | 3 | near_misses_mean | 5.3333 | 5.6667 | 0.3334 |
| ppo | perpendicular_traffic | 3 | 3 | snqi_mean | -0.4478 | -0.1945 | 0.2533 |
| ppo | perpendicular_traffic | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.5224 | -0.4776 |
| ppo | perpendicular_traffic | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | perpendicular_traffic | 3 | 3 | comfort_exposure_mean | 0.0081 | 0.0073 | -0.0008 |
| ppo | robot_overtaking | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| ppo | robot_overtaking | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| ppo | robot_overtaking | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | robot_overtaking | 3 | 3 | near_misses_mean | 7.3333 | 7.6667 | 0.3334 |
| ppo | robot_overtaking | 3 | 3 | snqi_mean | -0.2848 | -0.0590 | 0.2258 |
| ppo | robot_overtaking | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6542 | -0.3458 |
| ppo | robot_overtaking | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | robot_overtaking | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | crowd_navigation | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| prediction_planner | crowd_navigation | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| prediction_planner | crowd_navigation | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | crowd_navigation | 3 | 3 | near_misses_mean | 24.3333 | 79.6667 | 55.3334 |
| prediction_planner | crowd_navigation | 3 | 3 | snqi_mean | -0.4837 | -0.2552 | 0.2285 |
| prediction_planner | crowd_navigation | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6558 | -0.3442 |
| prediction_planner | crowd_navigation | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | crowd_navigation | 3 | 3 | comfort_exposure_mean | 0.0833 | 0.0491 | -0.0342 |
| prediction_planner | intersection_no_gesture | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| prediction_planner | intersection_no_gesture | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| prediction_planner | intersection_no_gesture | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | intersection_no_gesture | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | intersection_no_gesture | 3 | 3 | snqi_mean | -0.0949 | 0.1138 | 0.2087 |
| prediction_planner | intersection_no_gesture | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.8075 | -0.1925 |
| prediction_planner | intersection_no_gesture | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | intersection_no_gesture | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | intersection_proceed | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| prediction_planner | intersection_proceed | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| prediction_planner | intersection_proceed | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | intersection_proceed | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | intersection_proceed | 3 | 3 | snqi_mean | -0.0949 | 0.1138 | 0.2087 |
| prediction_planner | intersection_proceed | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.8075 | -0.1925 |
| prediction_planner | intersection_proceed | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | intersection_proceed | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | intersection_wait | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| prediction_planner | intersection_wait | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| prediction_planner | intersection_wait | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | intersection_wait | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | intersection_wait | 3 | 3 | snqi_mean | -0.0949 | 0.1138 | 0.2087 |
| prediction_planner | intersection_wait | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.8075 | -0.1925 |
| prediction_planner | intersection_wait | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | intersection_wait | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | pedestrian_overtaking | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| prediction_planner | pedestrian_overtaking | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| prediction_planner | pedestrian_overtaking | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | pedestrian_overtaking | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | pedestrian_overtaking | 3 | 3 | snqi_mean | -0.1149 | 0.0911 | 0.2060 |
| prediction_planner | pedestrian_overtaking | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.8994 | -0.1006 |
| prediction_planner | pedestrian_overtaking | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | pedestrian_overtaking | 3 | 3 | comfort_exposure_mean | 0.0667 | 0.0334 | -0.0333 |
| prediction_planner | perpendicular_traffic | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| prediction_planner | perpendicular_traffic | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| prediction_planner | perpendicular_traffic | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | perpendicular_traffic | 3 | 3 | near_misses_mean | 21.6667 | 40.6667 | 19.0000 |
| prediction_planner | perpendicular_traffic | 3 | 3 | snqi_mean | -0.3007 | -0.0898 | 0.2109 |
| prediction_planner | perpendicular_traffic | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.7869 | -0.2131 |
| prediction_planner | perpendicular_traffic | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | perpendicular_traffic | 3 | 3 | comfort_exposure_mean | 0.0081 | 0.0049 | -0.0032 |
| sacadrl | crowd_navigation | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| sacadrl | crowd_navigation | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| sacadrl | crowd_navigation | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| sacadrl | crowd_navigation | 3 | 3 | near_misses_mean | 0.0000 | 4.6667 | 4.6667 |
| sacadrl | crowd_navigation | 3 | 3 | snqi_mean | -0.1748 | -0.0239 | 0.1509 |
| sacadrl | crowd_navigation | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.7190 | -0.2810 |
| sacadrl | crowd_navigation | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| sacadrl | crowd_navigation | 3 | 3 | comfort_exposure_mean | 0.0826 | 0.0439 | -0.0387 |
| sacadrl | exiting_elevator | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| sacadrl | exiting_elevator | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| sacadrl | exiting_elevator | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| sacadrl | exiting_elevator | 3 | 3 | near_misses_mean | 21.0000 | 21.0000 | 0.0000 |
| sacadrl | exiting_elevator | 3 | 3 | snqi_mean | -0.4572 | -0.2025 | 0.2547 |
| sacadrl | exiting_elevator | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.3926 | -0.6074 |
| sacadrl | exiting_elevator | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| sacadrl | exiting_elevator | 3 | 3 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | pedestrian_overtaking | 3 | 3 | success_mean | 0.0000 | 1.0000 | 1.0000 |
| socnav_sampling | pedestrian_overtaking | 3 | 3 | unfinished_mean | 1.0000 | 0.0000 | -1.0000 |
| socnav_sampling | pedestrian_overtaking | 3 | 3 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | pedestrian_overtaking | 3 | 3 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | pedestrian_overtaking | 3 | 3 | snqi_mean | -0.1192 | 0.1056 | 0.2248 |
| socnav_sampling | pedestrian_overtaking | 3 | 3 | time_to_goal_norm_mean | 1.0000 | 0.6922 | -0.3078 |
| socnav_sampling | pedestrian_overtaking | 3 | 3 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | pedestrian_overtaking | 3 | 3 | comfort_exposure_mean | 0.0800 | 0.0517 | -0.0283 |
| ppo | cross_trap | 9 | 9 | success_mean | 0.0000 | 0.8889 | 0.8889 |
| ppo | cross_trap | 9 | 9 | unfinished_mean | 1.0000 | 0.1111 | -0.8889 |
| ppo | cross_trap | 9 | 9 | collisions_mean | 0.1111 | 0.1111 | 0.0000 |
| ppo | cross_trap | 9 | 9 | near_misses_mean | 7.0000 | 20.0000 | 13.0000 |
| ppo | cross_trap | 9 | 9 | snqi_mean | -0.3705 | -0.2720 | 0.0985 |
| ppo | cross_trap | 9 | 9 | time_to_goal_norm_mean | 1.0000 | 0.5040 | -0.4960 |
| ppo | cross_trap | 9 | 9 | path_efficiency_mean | 0.9681 | 0.7406 | -0.2275 |
| ppo | cross_trap | 9 | 9 | comfort_exposure_mean | 0.0782 | 0.0713 | -0.0069 |

