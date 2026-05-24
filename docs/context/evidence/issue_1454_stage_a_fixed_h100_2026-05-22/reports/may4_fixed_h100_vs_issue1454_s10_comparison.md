# Camera-Ready Campaign Comparison

- Base campaign: `paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217`
- Candidate campaign: `issue1454-s10-fixed-h100`
- Reproducibility verdict: `drift_detected`

## Planner Deltas

| planner | base_status | candidate_status | base_episodes | candidate_episodes | exact_match | metric | base | candidate | delta(candidate-base) |
|---|---|---|---:|---:|---|---|---:|---:|---:|
| goal | ok | ok | 144 | 480 | no | success_mean | 0.0139 | 0.0063 | -0.0076 |
| goal | ok | ok | 144 | 480 | no | unfinished_mean | 0.9861 | 0.9937 | 0.0076 |
| goal | ok | ok | 144 | 480 | no | collisions_mean | 0.2361 | 0.2396 | 0.0035 |
| goal | ok | ok | 144 | 480 | no | near_misses_mean | 2.9097 | 2.1396 | -0.7701 |
| goal | ok | ok | 144 | 480 | no | snqi_mean | -0.1656 | -0.1838 | -0.0182 |
| goal | ok | ok | 144 | 480 | no | time_to_goal_norm_mean | 0.9981 | 0.9995 | 0.0014 |
| goal | ok | ok | 144 | 480 | no | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| goal | ok | ok | 144 | 480 | no | comfort_exposure_mean | 0.0297 | 0.0298 | 0.0001 |
| goal | ok | ok | 144 | 480 | no | jerk_mean | 0.0092 | 0.0109 | 0.0017 |
| orca | ok | ok | 144 | 480 | no | success_mean | 0.1806 | 0.1812 | 0.0006 |
| orca | ok | ok | 144 | 480 | no | unfinished_mean | 0.8194 | 0.8188 | -0.0006 |
| orca | ok | ok | 144 | 480 | no | collisions_mean | 0.0347 | 0.0604 | 0.0257 |
| orca | ok | ok | 144 | 480 | no | near_misses_mean | 4.9097 | 4.7479 | -0.1618 |
| orca | ok | ok | 144 | 480 | no | snqi_mean | -0.2589 | -0.2497 | 0.0092 |
| orca | ok | ok | 144 | 480 | no | time_to_goal_norm_mean | 0.9650 | 0.9658 | 0.0008 |
| orca | ok | ok | 144 | 480 | no | path_efficiency_mean | 1.0000 | 0.9996 | -0.0004 |
| orca | ok | ok | 144 | 480 | no | comfort_exposure_mean | 0.0310 | 0.0314 | 0.0004 |
| orca | ok | ok | 144 | 480 | no | jerk_mean | 0.1718 | 0.1760 | 0.0042 |
| ppo | ok | ok | 144 | 480 | no | success_mean | 0.2500 | 0.2250 | -0.0250 |
| ppo | ok | ok | 144 | 480 | no | unfinished_mean | 0.7500 | 0.7750 | 0.0250 |
| ppo | ok | ok | 144 | 480 | no | collisions_mean | 0.0903 | 0.1229 | 0.0326 |
| ppo | ok | ok | 144 | 480 | no | near_misses_mean | 3.3542 | 3.9750 | 0.6208 |
| ppo | ok | ok | 144 | 480 | no | snqi_mean | -0.3060 | -0.3280 | -0.0220 |
| ppo | ok | ok | 144 | 480 | no | time_to_goal_norm_mean | 0.9306 | 0.9334 | 0.0028 |
| ppo | ok | ok | 144 | 480 | no | path_efficiency_mean | 0.9866 | 0.9772 | -0.0094 |
| ppo | ok | ok | 144 | 480 | no | comfort_exposure_mean | 0.0292 | 0.0282 | -0.0010 |
| ppo | ok | ok | 144 | 480 | no | jerk_mean | 0.4629 | 0.4435 | -0.0194 |
| prediction_planner | ok | ok | 144 | 480 | no | success_mean | 0.0694 | 0.0625 | -0.0069 |
| prediction_planner | ok | ok | 144 | 480 | no | unfinished_mean | 0.9306 | 0.9375 | 0.0069 |
| prediction_planner | ok | ok | 144 | 480 | no | collisions_mean | 0.2083 | 0.2229 | 0.0146 |
| prediction_planner | ok | ok | 144 | 480 | no | near_misses_mean | 8.3681 | 7.2667 | -1.1014 |
| prediction_planner | ok | ok | 144 | 480 | no | snqi_mean | -0.1945 | -0.2076 | -0.0131 |
| prediction_planner | ok | ok | 144 | 480 | no | time_to_goal_norm_mean | 0.9867 | 0.9869 | 0.0002 |
| prediction_planner | ok | ok | 144 | 480 | no | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | ok | ok | 144 | 480 | no | comfort_exposure_mean | 0.0337 | 0.0321 | -0.0016 |
| prediction_planner | ok | ok | 144 | 480 | no | jerk_mean | 0.0835 | 0.0858 | 0.0023 |
| sacadrl | ok | ok | 144 | 480 | no | success_mean | 0.0000 | 0.0104 | 0.0104 |
| sacadrl | ok | ok | 144 | 480 | no | unfinished_mean | 1.0000 | 0.9896 | -0.0104 |
| sacadrl | ok | ok | 144 | 480 | no | collisions_mean | 0.3889 | 0.3792 | -0.0097 |
| sacadrl | ok | ok | 144 | 480 | no | near_misses_mean | 3.5694 | 2.4438 | -1.1256 |
| sacadrl | ok | ok | 144 | 480 | no | snqi_mean | -0.2834 | -0.3111 | -0.0277 |
| sacadrl | ok | ok | 144 | 480 | no | time_to_goal_norm_mean | 1.0000 | 0.9995 | -0.0005 |
| sacadrl | ok | ok | 144 | 480 | no | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| sacadrl | ok | ok | 144 | 480 | no | comfort_exposure_mean | 0.0302 | 0.0285 | -0.0017 |
| sacadrl | ok | ok | 144 | 480 | no | jerk_mean | 0.0763 | 0.0779 | 0.0016 |
| social_force | ok | ok | 144 | 480 | no | success_mean | 0.0000 | 0.0000 | 0.0000 |
| social_force | ok | ok | 144 | 480 | no | unfinished_mean | 1.0000 | 1.0000 | 0.0000 |
| social_force | ok | ok | 144 | 480 | no | collisions_mean | 0.2083 | 0.2250 | 0.0167 |
| social_force | ok | ok | 144 | 480 | no | near_misses_mean | 2.3264 | 2.5583 | 0.2319 |
| social_force | ok | ok | 144 | 480 | no | snqi_mean | -0.8535 | -0.8704 | -0.0169 |
| social_force | ok | ok | 144 | 480 | no | time_to_goal_norm_mean | 1.0000 | 1.0000 | 0.0000 |
| social_force | ok | ok | 144 | 480 | no | path_efficiency_mean | 0.9362 | 0.9301 | -0.0061 |
| social_force | ok | ok | 144 | 480 | no | comfort_exposure_mean | 0.0435 | 0.0402 | -0.0033 |
| social_force | ok | ok | 144 | 480 | no | jerk_mean | 0.4937 | 0.4774 | -0.0163 |
| socnav_bench | failed | failed | 0 | 0 | yes | N/A | N/A | N/A | N/A |
| socnav_sampling | ok | ok | 144 | 480 | no | success_mean | 0.1736 | 0.1604 | -0.0132 |
| socnav_sampling | ok | ok | 144 | 480 | no | unfinished_mean | 0.8264 | 0.8396 | 0.0132 |
| socnav_sampling | ok | ok | 144 | 480 | no | collisions_mean | 0.5278 | 0.4792 | -0.0486 |
| socnav_sampling | ok | ok | 144 | 480 | no | near_misses_mean | 1.4097 | 1.4083 | -0.0014 |
| socnav_sampling | ok | ok | 144 | 480 | no | snqi_mean | -0.1390 | -0.1923 | -0.0533 |
| socnav_sampling | ok | ok | 144 | 480 | no | time_to_goal_norm_mean | 0.9528 | 0.9517 | -0.0011 |
| socnav_sampling | ok | ok | 144 | 480 | no | path_efficiency_mean | 0.9918 | 0.9901 | -0.0017 |
| socnav_sampling | ok | ok | 144 | 480 | no | comfort_exposure_mean | 0.0363 | 0.0348 | -0.0015 |
| socnav_sampling | ok | ok | 144 | 480 | no | jerk_mean | 0.0724 | 0.0743 | 0.0019 |

## Coverage Gaps

- No planner coverage gaps.

## Reproducibility

- Status: `drift_detected`
- Exact-match planners: `socnav_bench`
- Mismatched planners: `goal, orca, ppo, prediction_planner, sacadrl, social_force, socnav_sampling`

## Scenario Deltas

- Complete machine-readable deltas are in the JSON artifact; showing up to 40 rows sorted by absolute success delta.

| planner_key | scenario_family | scenario_id | base_episodes | candidate_episodes | metric | base | candidate | delta(candidate-base) |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| sacadrl | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | success_mean | 0.0000 | 0.5000 | 0.5000 |
| sacadrl | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | unfinished_mean | 1.0000 | 0.5000 | -0.5000 |
| sacadrl | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | collisions_mean | 0.0000 | 0.1000 | 0.1000 |
| sacadrl | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | near_misses_mean | 21.0000 | 18.0000 | -3.0000 |
| sacadrl | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | snqi_mean | -0.4572 | -0.3200 | 0.1372 |
| sacadrl | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9750 | -0.0250 |
| sacadrl | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| sacadrl | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | success_mean | 0.0000 | 0.4000 | 0.4000 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | unfinished_mean | 1.0000 | 0.6000 | -0.4000 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | near_misses_mean | 11.0000 | 10.0000 | -1.0000 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | snqi_mean | -0.6476 | -0.6443 | 0.0033 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9820 | -0.0180 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| goal | entering_elevator | francis2023_entering_elevator | 3 | 10 | success_mean | 0.6667 | 0.3000 | -0.3667 |
| goal | entering_elevator | francis2023_entering_elevator | 3 | 10 | unfinished_mean | 0.3333 | 0.7000 | 0.3667 |
| goal | entering_elevator | francis2023_entering_elevator | 3 | 10 | collisions_mean | 0.3333 | 0.3000 | -0.0333 |
| goal | entering_elevator | francis2023_entering_elevator | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| goal | entering_elevator | francis2023_entering_elevator | 3 | 10 | snqi_mean | 0.0409 | -0.0671 | -0.1080 |
| goal | entering_elevator | francis2023_entering_elevator | 3 | 10 | time_to_goal_norm_mean | 0.9067 | 0.9780 | 0.0713 |
| goal | entering_elevator | francis2023_entering_elevator | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| goal | entering_elevator | francis2023_entering_elevator | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | entering_elevator | francis2023_entering_elevator | 3 | 10 | success_mean | 0.3333 | 0.7000 | 0.3667 |
| orca | entering_elevator | francis2023_entering_elevator | 3 | 10 | unfinished_mean | 0.6667 | 0.3000 | -0.3667 |
| orca | entering_elevator | francis2023_entering_elevator | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | entering_elevator | francis2023_entering_elevator | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | entering_elevator | francis2023_entering_elevator | 3 | 10 | snqi_mean | -0.1581 | -0.1100 | 0.0481 |
| orca | entering_elevator | francis2023_entering_elevator | 3 | 10 | time_to_goal_norm_mean | 0.9367 | 0.8830 | -0.0537 |
| orca | entering_elevator | francis2023_entering_elevator | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | entering_elevator | francis2023_entering_elevator | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | doorway | classic_doorway_low | 3 | 10 | success_mean | 0.6667 | 0.3000 | -0.3667 |
| ppo | doorway | classic_doorway_low | 3 | 10 | unfinished_mean | 0.3333 | 0.7000 | 0.3667 |
| ppo | doorway | classic_doorway_low | 3 | 10 | collisions_mean | 0.0000 | 0.3000 | 0.3000 |
| ppo | doorway | classic_doorway_low | 3 | 10 | near_misses_mean | 9.6667 | 7.8000 | -1.8667 |
| ppo | doorway | classic_doorway_low | 3 | 10 | snqi_mean | -0.7953 | -0.9077 | -0.1124 |
| ppo | doorway | classic_doorway_low | 3 | 10 | time_to_goal_norm_mean | 0.9733 | 0.9230 | -0.0503 |
| ppo | doorway | classic_doorway_low | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | doorway | classic_doorway_low | 3 | 10 | comfort_exposure_mean | 0.0354 | 0.0235 | -0.0119 |
| ppo | group_crossing | classic_group_crossing_low | 3 | 10 | success_mean | 0.6667 | 1.0000 | 0.3333 |
| ppo | group_crossing | classic_group_crossing_low | 3 | 10 | unfinished_mean | 0.3333 | 0.0000 | -0.3333 |
| ppo | group_crossing | classic_group_crossing_low | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | group_crossing | classic_group_crossing_low | 3 | 10 | near_misses_mean | 0.0000 | 2.1000 | 2.1000 |
| ppo | group_crossing | classic_group_crossing_low | 3 | 10 | snqi_mean | -0.1736 | -0.1151 | 0.0585 |
| ppo | group_crossing | classic_group_crossing_low | 3 | 10 | time_to_goal_norm_mean | 0.9033 | 0.8930 | -0.0103 |
| ppo | group_crossing | classic_group_crossing_low | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | group_crossing | classic_group_crossing_low | 3 | 10 | comfort_exposure_mean | 0.0078 | 0.0083 | 0.0005 |
| orca | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| orca | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| orca | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | snqi_mean | -0.0311 | -0.0949 | -0.0638 |
| orca | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| orca | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| orca | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| orca | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | snqi_mean | -0.0311 | -0.0949 | -0.0638 |
| orca | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| orca | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_wait | francis2023_intersection_wait | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| orca | intersection_wait | francis2023_intersection_wait | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| orca | intersection_wait | francis2023_intersection_wait | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_wait | francis2023_intersection_wait | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_wait | francis2023_intersection_wait | 3 | 10 | snqi_mean | -0.0311 | -0.0949 | -0.0638 |
| orca | intersection_wait | francis2023_intersection_wait | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| orca | intersection_wait | francis2023_intersection_wait | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | intersection_wait | francis2023_intersection_wait | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| ppo | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| ppo | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | snqi_mean | -0.2399 | -0.2975 | -0.0576 |
| ppo | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| ppo | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| ppo | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| ppo | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | snqi_mean | -0.2392 | -0.3041 | -0.0649 |
| ppo | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| ppo | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_wait | francis2023_intersection_wait | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| ppo | intersection_wait | francis2023_intersection_wait | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| ppo | intersection_wait | francis2023_intersection_wait | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_wait | francis2023_intersection_wait | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_wait | francis2023_intersection_wait | 3 | 10 | snqi_mean | -0.2392 | -0.3041 | -0.0649 |
| ppo | intersection_wait | francis2023_intersection_wait | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| ppo | intersection_wait | francis2023_intersection_wait | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | intersection_wait | francis2023_intersection_wait | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | group_crossing | classic_group_crossing_medium | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| prediction_planner | group_crossing | classic_group_crossing_medium | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| prediction_planner | group_crossing | classic_group_crossing_medium | 3 | 10 | collisions_mean | 0.3333 | 0.1000 | -0.2333 |
| prediction_planner | group_crossing | classic_group_crossing_medium | 3 | 10 | near_misses_mean | 3.0000 | 10.7000 | 7.7000 |
| prediction_planner | group_crossing | classic_group_crossing_medium | 3 | 10 | snqi_mean | -0.0982 | -0.2438 | -0.1456 |
| prediction_planner | group_crossing | classic_group_crossing_medium | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| prediction_planner | group_crossing | classic_group_crossing_medium | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | group_crossing | classic_group_crossing_medium | 3 | 10 | comfort_exposure_mean | 0.0271 | 0.0273 | 0.0002 |
| socnav_sampling | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| socnav_sampling | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| socnav_sampling | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | snqi_mean | -0.0311 | -0.0949 | -0.0638 |
| socnav_sampling | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| socnav_sampling | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | intersection_no_gesture | francis2023_intersection_no_gesture | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| socnav_sampling | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| socnav_sampling | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | snqi_mean | -0.0311 | -0.0949 | -0.0638 |
| socnav_sampling | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| socnav_sampling | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | intersection_proceed | francis2023_intersection_proceed | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_wait | francis2023_intersection_wait | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| socnav_sampling | intersection_wait | francis2023_intersection_wait | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| socnav_sampling | intersection_wait | francis2023_intersection_wait | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_wait | francis2023_intersection_wait | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_wait | francis2023_intersection_wait | 3 | 10 | snqi_mean | -0.0311 | -0.0949 | -0.0638 |
| socnav_sampling | intersection_wait | francis2023_intersection_wait | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| socnav_sampling | intersection_wait | francis2023_intersection_wait | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | intersection_wait | francis2023_intersection_wait | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | group_crossing | classic_group_crossing_high | 3 | 10 | success_mean | 1.0000 | 0.7000 | -0.3000 |
| ppo | group_crossing | classic_group_crossing_high | 3 | 10 | unfinished_mean | 0.0000 | 0.3000 | 0.3000 |
| ppo | group_crossing | classic_group_crossing_high | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | group_crossing | classic_group_crossing_high | 3 | 10 | near_misses_mean | 3.3333 | 6.2000 | 2.8667 |
| ppo | group_crossing | classic_group_crossing_high | 3 | 10 | snqi_mean | -0.1251 | -0.2835 | -0.1584 |
| ppo | group_crossing | classic_group_crossing_high | 3 | 10 | time_to_goal_norm_mean | 0.8167 | 0.9230 | 0.1063 |
| ppo | group_crossing | classic_group_crossing_high | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | group_crossing | classic_group_crossing_high | 3 | 10 | comfort_exposure_mean | 0.0132 | 0.0222 | 0.0090 |
| prediction_planner | entering_room | francis2023_entering_room | 3 | 10 | success_mean | 0.6667 | 0.4000 | -0.2667 |
| prediction_planner | entering_room | francis2023_entering_room | 3 | 10 | unfinished_mean | 0.3333 | 0.6000 | 0.2667 |
| prediction_planner | entering_room | francis2023_entering_room | 3 | 10 | collisions_mean | 0.3333 | 0.3000 | -0.0333 |
| prediction_planner | entering_room | francis2023_entering_room | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | entering_room | francis2023_entering_room | 3 | 10 | snqi_mean | 0.0378 | -0.0490 | -0.0868 |
| prediction_planner | entering_room | francis2023_entering_room | 3 | 10 | time_to_goal_norm_mean | 0.9400 | 0.9840 | 0.0440 |
| prediction_planner | entering_room | francis2023_entering_room | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | entering_room | francis2023_entering_room | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | circular_crossing | francis2023_circular_crossing | 3 | 10 | success_mean | 0.6667 | 0.9000 | 0.2333 |
| orca | circular_crossing | francis2023_circular_crossing | 3 | 10 | unfinished_mean | 0.3333 | 0.1000 | -0.2333 |
| orca | circular_crossing | francis2023_circular_crossing | 3 | 10 | collisions_mean | 0.3333 | 0.1000 | -0.2333 |
| orca | circular_crossing | francis2023_circular_crossing | 3 | 10 | near_misses_mean | 1.0000 | 0.3000 | -0.7000 |
| orca | circular_crossing | francis2023_circular_crossing | 3 | 10 | snqi_mean | -0.0403 | 0.0385 | 0.0788 |
| orca | circular_crossing | francis2023_circular_crossing | 3 | 10 | time_to_goal_norm_mean | 0.7667 | 0.7160 | -0.0507 |
| orca | circular_crossing | francis2023_circular_crossing | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | circular_crossing | francis2023_circular_crossing | 3 | 10 | comfort_exposure_mean | 0.1975 | 0.0882 | -0.1093 |
| ppo | circular_crossing | francis2023_circular_crossing | 3 | 10 | success_mean | 0.6667 | 0.9000 | 0.2333 |
| ppo | circular_crossing | francis2023_circular_crossing | 3 | 10 | unfinished_mean | 0.3333 | 0.1000 | -0.2333 |
| ppo | circular_crossing | francis2023_circular_crossing | 3 | 10 | collisions_mean | 0.3333 | 0.1000 | -0.2333 |
| ppo | circular_crossing | francis2023_circular_crossing | 3 | 10 | near_misses_mean | 0.6667 | 0.3000 | -0.3667 |
| ppo | circular_crossing | francis2023_circular_crossing | 3 | 10 | snqi_mean | -0.1323 | -0.1229 | 0.0094 |
| ppo | circular_crossing | francis2023_circular_crossing | 3 | 10 | time_to_goal_norm_mean | 0.6967 | 0.6120 | -0.0847 |
| ppo | circular_crossing | francis2023_circular_crossing | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | circular_crossing | francis2023_circular_crossing | 3 | 10 | comfort_exposure_mean | 0.1964 | 0.0920 | -0.1044 |
| prediction_planner | circular_crossing | francis2023_circular_crossing | 3 | 10 | success_mean | 0.6667 | 0.9000 | 0.2333 |
| prediction_planner | circular_crossing | francis2023_circular_crossing | 3 | 10 | unfinished_mean | 0.3333 | 0.1000 | -0.2333 |
| prediction_planner | circular_crossing | francis2023_circular_crossing | 3 | 10 | collisions_mean | 0.3333 | 0.1000 | -0.2333 |
| prediction_planner | circular_crossing | francis2023_circular_crossing | 3 | 10 | near_misses_mean | 1.3333 | 0.4000 | -0.9333 |
| prediction_planner | circular_crossing | francis2023_circular_crossing | 3 | 10 | snqi_mean | -0.0217 | 0.0542 | 0.0759 |
| prediction_planner | circular_crossing | francis2023_circular_crossing | 3 | 10 | time_to_goal_norm_mean | 0.8267 | 0.7720 | -0.0547 |
| prediction_planner | circular_crossing | francis2023_circular_crossing | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | circular_crossing | francis2023_circular_crossing | 3 | 10 | comfort_exposure_mean | 0.1925 | 0.0864 | -0.1061 |
| socnav_sampling | circular_crossing | francis2023_circular_crossing | 3 | 10 | success_mean | 0.6667 | 0.9000 | 0.2333 |
| socnav_sampling | circular_crossing | francis2023_circular_crossing | 3 | 10 | unfinished_mean | 0.3333 | 0.1000 | -0.2333 |
| socnav_sampling | circular_crossing | francis2023_circular_crossing | 3 | 10 | collisions_mean | 0.3333 | 0.1000 | -0.2333 |
| socnav_sampling | circular_crossing | francis2023_circular_crossing | 3 | 10 | near_misses_mean | 0.6667 | 0.3000 | -0.3667 |
| socnav_sampling | circular_crossing | francis2023_circular_crossing | 3 | 10 | snqi_mean | -0.0222 | 0.0447 | 0.0669 |
| socnav_sampling | circular_crossing | francis2023_circular_crossing | 3 | 10 | time_to_goal_norm_mean | 0.7367 | 0.6240 | -0.1127 |
| socnav_sampling | circular_crossing | francis2023_circular_crossing | 3 | 10 | path_efficiency_mean | 0.9907 | 1.0000 | 0.0093 |
| socnav_sampling | circular_crossing | francis2023_circular_crossing | 3 | 10 | comfort_exposure_mean | 0.1948 | 0.0912 | -0.1036 |
| ppo | doorway | classic_doorway_high | 3 | 10 | success_mean | 0.3333 | 0.1000 | -0.2333 |
| ppo | doorway | classic_doorway_high | 3 | 10 | unfinished_mean | 0.6667 | 0.9000 | 0.2333 |
| ppo | doorway | classic_doorway_high | 3 | 10 | collisions_mean | 0.3333 | 0.4000 | 0.0667 |
| ppo | doorway | classic_doorway_high | 3 | 10 | near_misses_mean | 17.0000 | 11.3000 | -5.7000 |
| ppo | doorway | classic_doorway_high | 3 | 10 | snqi_mean | -1.1632 | -1.0372 | 0.1260 |
| ppo | doorway | classic_doorway_high | 3 | 10 | time_to_goal_norm_mean | 0.9067 | 0.9700 | 0.0633 |
| ppo | doorway | classic_doorway_high | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | doorway | classic_doorway_high | 3 | 10 | comfort_exposure_mean | 0.0122 | 0.0114 | -0.0008 |
| orca | robot_crowding | francis2023_robot_crowding | 3 | 10 | success_mean | 0.0000 | 0.2000 | 0.2000 |
| orca | robot_crowding | francis2023_robot_crowding | 3 | 10 | unfinished_mean | 1.0000 | 0.8000 | -0.2000 |
| orca | robot_crowding | francis2023_robot_crowding | 3 | 10 | collisions_mean | 0.3333 | 0.4000 | 0.0667 |
| orca | robot_crowding | francis2023_robot_crowding | 3 | 10 | near_misses_mean | 29.0000 | 20.5000 | -8.5000 |
| orca | robot_crowding | francis2023_robot_crowding | 3 | 10 | snqi_mean | -1.2204 | -0.8207 | 0.3997 |
| orca | robot_crowding | francis2023_robot_crowding | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9770 | -0.0230 |
| orca | robot_crowding | francis2023_robot_crowding | 3 | 10 | path_efficiency_mean | 1.0000 | 0.9817 | -0.0183 |
| orca | robot_crowding | francis2023_robot_crowding | 3 | 10 | comfort_exposure_mean | 0.1807 | 0.2142 | 0.0335 |
| socnav_sampling | bottleneck | classic_bottleneck_low | 3 | 10 | success_mean | 0.0000 | 0.2000 | 0.2000 |
| socnav_sampling | bottleneck | classic_bottleneck_low | 3 | 10 | unfinished_mean | 1.0000 | 0.8000 | -0.2000 |
| socnav_sampling | bottleneck | classic_bottleneck_low | 3 | 10 | collisions_mean | 0.0000 | 0.1000 | 0.1000 |
| socnav_sampling | bottleneck | classic_bottleneck_low | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | bottleneck | classic_bottleneck_low | 3 | 10 | snqi_mean | -0.0949 | -0.0683 | 0.0266 |
| socnav_sampling | bottleneck | classic_bottleneck_low | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9970 | -0.0030 |
| socnav_sampling | bottleneck | classic_bottleneck_low | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | bottleneck | classic_bottleneck_low | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | group_crossing | classic_group_crossing_low | 3 | 10 | success_mean | 1.0000 | 0.8000 | -0.2000 |
| orca | group_crossing | classic_group_crossing_low | 3 | 10 | unfinished_mean | 0.0000 | 0.2000 | 0.2000 |
| orca | group_crossing | classic_group_crossing_low | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | group_crossing | classic_group_crossing_low | 3 | 10 | near_misses_mean | 0.0000 | 4.7000 | 4.7000 |
| orca | group_crossing | classic_group_crossing_low | 3 | 10 | snqi_mean | 0.1026 | -0.0642 | -0.1668 |
| orca | group_crossing | classic_group_crossing_low | 3 | 10 | time_to_goal_norm_mean | 0.8967 | 0.9110 | 0.0143 |
| orca | group_crossing | classic_group_crossing_low | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | group_crossing | classic_group_crossing_low | 3 | 10 | comfort_exposure_mean | 0.0077 | 0.0252 | 0.0175 |
| socnav_sampling | group_crossing | classic_group_crossing_medium | 3 | 10 | success_mean | 0.6667 | 0.5000 | -0.1667 |
| socnav_sampling | group_crossing | classic_group_crossing_medium | 3 | 10 | unfinished_mean | 0.3333 | 0.5000 | 0.1667 |
| socnav_sampling | group_crossing | classic_group_crossing_medium | 3 | 10 | collisions_mean | 0.3333 | 0.5000 | 0.1667 |
| socnav_sampling | group_crossing | classic_group_crossing_medium | 3 | 10 | near_misses_mean | 1.0000 | 2.1000 | 1.1000 |
| socnav_sampling | group_crossing | classic_group_crossing_medium | 3 | 10 | snqi_mean | -0.0127 | -0.1163 | -0.1036 |
| socnav_sampling | group_crossing | classic_group_crossing_medium | 3 | 10 | time_to_goal_norm_mean | 0.8867 | 0.9460 | 0.0593 |
| socnav_sampling | group_crossing | classic_group_crossing_medium | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | group_crossing | classic_group_crossing_medium | 3 | 10 | comfort_exposure_mean | 0.0319 | 0.0313 | -0.0006 |
| prediction_planner | doorway | classic_doorway_low | 3 | 10 | success_mean | 0.3333 | 0.2000 | -0.1333 |
| prediction_planner | doorway | classic_doorway_low | 3 | 10 | unfinished_mean | 0.6667 | 0.8000 | 0.1333 |
| prediction_planner | doorway | classic_doorway_low | 3 | 10 | collisions_mean | 0.3333 | 0.4000 | 0.0667 |
| prediction_planner | doorway | classic_doorway_low | 3 | 10 | near_misses_mean | 29.0000 | 23.1000 | -5.9000 |
| prediction_planner | doorway | classic_doorway_low | 3 | 10 | snqi_mean | -0.3326 | -0.4341 | -0.1015 |
| prediction_planner | doorway | classic_doorway_low | 3 | 10 | time_to_goal_norm_mean | 0.9367 | 0.9650 | 0.0283 |
| prediction_planner | doorway | classic_doorway_low | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | doorway | classic_doorway_low | 3 | 10 | comfort_exposure_mean | 0.0259 | 0.0321 | 0.0062 |
| prediction_planner | doorway | classic_doorway_medium | 3 | 10 | success_mean | 0.3333 | 0.2000 | -0.1333 |
| prediction_planner | doorway | classic_doorway_medium | 3 | 10 | unfinished_mean | 0.6667 | 0.8000 | 0.1333 |
| prediction_planner | doorway | classic_doorway_medium | 3 | 10 | collisions_mean | 0.3333 | 0.3000 | -0.0333 |
| prediction_planner | doorway | classic_doorway_medium | 3 | 10 | near_misses_mean | 16.3333 | 23.7000 | 7.3667 |
| prediction_planner | doorway | classic_doorway_medium | 3 | 10 | snqi_mean | -0.2693 | -0.3552 | -0.0859 |
| prediction_planner | doorway | classic_doorway_medium | 3 | 10 | time_to_goal_norm_mean | 0.9900 | 0.9820 | -0.0080 |
| prediction_planner | doorway | classic_doorway_medium | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | doorway | classic_doorway_medium | 3 | 10 | comfort_exposure_mean | 0.0134 | 0.0100 | -0.0034 |
| orca | bottleneck | classic_bottleneck_low | 3 | 10 | success_mean | 0.0000 | 0.1000 | 0.1000 |
| orca | bottleneck | classic_bottleneck_low | 3 | 10 | unfinished_mean | 1.0000 | 0.9000 | -0.1000 |
| orca | bottleneck | classic_bottleneck_low | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | bottleneck | classic_bottleneck_low | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | bottleneck | classic_bottleneck_low | 3 | 10 | snqi_mean | -0.0949 | -0.0765 | 0.0184 |
| orca | bottleneck | classic_bottleneck_low | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9980 | -0.0020 |
| orca | bottleneck | classic_bottleneck_low | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | bottleneck | classic_bottleneck_low | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 10 | success_mean | 0.0000 | 0.1000 | 0.1000 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 10 | unfinished_mean | 1.0000 | 0.9000 | -0.1000 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 10 | snqi_mean | -0.2437 | -0.2190 | 0.0247 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9960 | -0.0040 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | bottleneck | classic_bottleneck_low | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 10 | success_mean | 0.0000 | 0.1000 | 0.1000 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 10 | unfinished_mean | 1.0000 | 0.9000 | -0.1000 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 10 | near_misses_mean | 5.0000 | 6.0000 | 1.0000 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 10 | snqi_mean | -0.3248 | -0.3191 | 0.0057 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9970 | -0.0030 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | bottleneck | classic_bottleneck_medium | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | join_group | francis2023_join_group | 3 | 10 | success_mean | 0.0000 | 0.1000 | 0.1000 |
| prediction_planner | join_group | francis2023_join_group | 3 | 10 | unfinished_mean | 1.0000 | 0.9000 | -0.1000 |
| prediction_planner | join_group | francis2023_join_group | 3 | 10 | collisions_mean | 0.3333 | 0.4000 | 0.0667 |
| prediction_planner | join_group | francis2023_join_group | 3 | 10 | near_misses_mean | 25.6667 | 23.4000 | -2.2667 |
| prediction_planner | join_group | francis2023_join_group | 3 | 10 | snqi_mean | -0.3726 | -0.3866 | -0.0140 |
| prediction_planner | join_group | francis2023_join_group | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9950 | -0.0050 |
| prediction_planner | join_group | francis2023_join_group | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | join_group | francis2023_join_group | 3 | 10 | comfort_exposure_mean | 0.0655 | 0.0570 | -0.0085 |
| prediction_planner | leave_group | francis2023_leave_group | 3 | 10 | success_mean | 0.0000 | 0.1000 | 0.1000 |
| prediction_planner | leave_group | francis2023_leave_group | 3 | 10 | unfinished_mean | 1.0000 | 0.9000 | -0.1000 |
| prediction_planner | leave_group | francis2023_leave_group | 3 | 10 | collisions_mean | 0.3333 | 0.6000 | 0.2667 |
| prediction_planner | leave_group | francis2023_leave_group | 3 | 10 | near_misses_mean | 48.3333 | 37.1000 | -11.2333 |
| prediction_planner | leave_group | francis2023_leave_group | 3 | 10 | snqi_mean | -0.4519 | -0.4802 | -0.0283 |
| prediction_planner | leave_group | francis2023_leave_group | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9920 | -0.0080 |
| prediction_planner | leave_group | francis2023_leave_group | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | leave_group | francis2023_leave_group | 3 | 10 | comfort_exposure_mean | 0.1190 | 0.0950 | -0.0240 |
| socnav_sampling | doorway | classic_doorway_medium | 3 | 10 | success_mean | 0.0000 | 0.1000 | 0.1000 |
| socnav_sampling | doorway | classic_doorway_medium | 3 | 10 | unfinished_mean | 1.0000 | 0.9000 | -0.1000 |
| socnav_sampling | doorway | classic_doorway_medium | 3 | 10 | collisions_mean | 1.0000 | 0.9000 | -0.1000 |
| socnav_sampling | doorway | classic_doorway_medium | 3 | 10 | near_misses_mean | 1.0000 | 2.2000 | 1.2000 |
| socnav_sampling | doorway | classic_doorway_medium | 3 | 10 | snqi_mean | -0.1891 | -0.2965 | -0.1074 |
| socnav_sampling | doorway | classic_doorway_medium | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9490 | -0.0510 |
| socnav_sampling | doorway | classic_doorway_medium | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | doorway | classic_doorway_medium | 3 | 10 | comfort_exposure_mean | 0.0159 | 0.0244 | 0.0085 |
| orca | group_crossing | classic_group_crossing_medium | 3 | 10 | success_mean | 1.0000 | 0.9000 | -0.1000 |
| orca | group_crossing | classic_group_crossing_medium | 3 | 10 | unfinished_mean | 0.0000 | 0.1000 | 0.1000 |
| orca | group_crossing | classic_group_crossing_medium | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | group_crossing | classic_group_crossing_medium | 3 | 10 | near_misses_mean | 2.3333 | 4.2000 | 1.8667 |
| orca | group_crossing | classic_group_crossing_medium | 3 | 10 | snqi_mean | 0.0572 | -0.0424 | -0.0996 |
| orca | group_crossing | classic_group_crossing_medium | 3 | 10 | time_to_goal_norm_mean | 0.8767 | 0.9170 | 0.0403 |
| orca | group_crossing | classic_group_crossing_medium | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | group_crossing | classic_group_crossing_medium | 3 | 10 | comfort_exposure_mean | 0.0327 | 0.0344 | 0.0017 |
| ppo | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | success_mean | 1.0000 | 0.9000 | -0.1000 |
| ppo | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | unfinished_mean | 0.0000 | 0.1000 | 0.1000 |
| ppo | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | collisions_mean | 0.0000 | 0.1000 | 0.1000 |
| ppo | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | near_misses_mean | 11.0000 | 10.5000 | -0.5000 |
| ppo | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | snqi_mean | -0.2959 | -0.3286 | -0.0327 |
| ppo | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | time_to_goal_norm_mean | 0.5467 | 0.5690 | 0.0223 |
| ppo | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | success_mean | 0.3333 | 0.4000 | 0.0667 |
| prediction_planner | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | unfinished_mean | 0.6667 | 0.6000 | -0.0667 |
| prediction_planner | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | collisions_mean | 0.3333 | 0.3000 | -0.0333 |
| prediction_planner | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | near_misses_mean | 27.0000 | 24.5000 | -2.5000 |
| prediction_planner | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | snqi_mean | -0.2445 | -0.2630 | -0.0185 |
| prediction_planner | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | time_to_goal_norm_mean | 0.9733 | 0.9380 | -0.0353 |
| prediction_planner | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | exiting_elevator | francis2023_exiting_elevator | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | group_crossing | classic_group_crossing_high | 3 | 10 | success_mean | 0.3333 | 0.4000 | 0.0667 |
| socnav_sampling | group_crossing | classic_group_crossing_high | 3 | 10 | unfinished_mean | 0.6667 | 0.6000 | -0.0667 |
| socnav_sampling | group_crossing | classic_group_crossing_high | 3 | 10 | collisions_mean | 0.6667 | 0.6000 | -0.0667 |
| socnav_sampling | group_crossing | classic_group_crossing_high | 3 | 10 | near_misses_mean | 1.3333 | 4.9000 | 3.5667 |
| socnav_sampling | group_crossing | classic_group_crossing_high | 3 | 10 | snqi_mean | -0.0792 | -0.2282 | -0.1490 |
| socnav_sampling | group_crossing | classic_group_crossing_high | 3 | 10 | time_to_goal_norm_mean | 0.9600 | 0.9330 | -0.0270 |
| socnav_sampling | group_crossing | classic_group_crossing_high | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | group_crossing | classic_group_crossing_high | 3 | 10 | comfort_exposure_mean | 0.0304 | 0.0393 | 0.0089 |
| socnav_sampling | group_crossing | classic_group_crossing_low | 3 | 10 | success_mean | 0.6667 | 0.6000 | -0.0667 |
| socnav_sampling | group_crossing | classic_group_crossing_low | 3 | 10 | unfinished_mean | 0.3333 | 0.4000 | 0.0667 |
| socnav_sampling | group_crossing | classic_group_crossing_low | 3 | 10 | collisions_mean | 0.3333 | 0.4000 | 0.0667 |
| socnav_sampling | group_crossing | classic_group_crossing_low | 3 | 10 | near_misses_mean | 0.0000 | 0.3000 | 0.3000 |
| socnav_sampling | group_crossing | classic_group_crossing_low | 3 | 10 | snqi_mean | 0.0228 | -0.0446 | -0.0674 |
| socnav_sampling | group_crossing | classic_group_crossing_low | 3 | 10 | time_to_goal_norm_mean | 0.9033 | 0.9240 | 0.0207 |
| socnav_sampling | group_crossing | classic_group_crossing_low | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | group_crossing | classic_group_crossing_low | 3 | 10 | comfort_exposure_mean | 0.0078 | 0.0083 | 0.0005 |

## Scenario Family Deltas

- Complete machine-readable deltas are in the JSON artifact; showing up to 40 rows sorted by absolute success delta.

| planner_key | scenario_family | base_episodes | candidate_episodes | metric | base | candidate | delta(candidate-base) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| sacadrl | exiting_elevator | 3 | 10 | success_mean | 0.0000 | 0.5000 | 0.5000 |
| sacadrl | exiting_elevator | 3 | 10 | unfinished_mean | 1.0000 | 0.5000 | -0.5000 |
| sacadrl | exiting_elevator | 3 | 10 | collisions_mean | 0.0000 | 0.1000 | 0.1000 |
| sacadrl | exiting_elevator | 3 | 10 | near_misses_mean | 21.0000 | 18.0000 | -3.0000 |
| sacadrl | exiting_elevator | 3 | 10 | snqi_mean | -0.4572 | -0.3200 | 0.1372 |
| sacadrl | exiting_elevator | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9750 | -0.0250 |
| sacadrl | exiting_elevator | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| sacadrl | exiting_elevator | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | exiting_elevator | 3 | 10 | success_mean | 0.0000 | 0.4000 | 0.4000 |
| orca | exiting_elevator | 3 | 10 | unfinished_mean | 1.0000 | 0.6000 | -0.4000 |
| orca | exiting_elevator | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | exiting_elevator | 3 | 10 | near_misses_mean | 11.0000 | 10.0000 | -1.0000 |
| orca | exiting_elevator | 3 | 10 | snqi_mean | -0.6476 | -0.6443 | 0.0033 |
| orca | exiting_elevator | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9820 | -0.0180 |
| orca | exiting_elevator | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | exiting_elevator | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| goal | entering_elevator | 3 | 10 | success_mean | 0.6667 | 0.3000 | -0.3667 |
| goal | entering_elevator | 3 | 10 | unfinished_mean | 0.3333 | 0.7000 | 0.3667 |
| goal | entering_elevator | 3 | 10 | collisions_mean | 0.3333 | 0.3000 | -0.0333 |
| goal | entering_elevator | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| goal | entering_elevator | 3 | 10 | snqi_mean | 0.0409 | -0.0671 | -0.1080 |
| goal | entering_elevator | 3 | 10 | time_to_goal_norm_mean | 0.9067 | 0.9780 | 0.0713 |
| goal | entering_elevator | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| goal | entering_elevator | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | entering_elevator | 3 | 10 | success_mean | 0.3333 | 0.7000 | 0.3667 |
| orca | entering_elevator | 3 | 10 | unfinished_mean | 0.6667 | 0.3000 | -0.3667 |
| orca | entering_elevator | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | entering_elevator | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | entering_elevator | 3 | 10 | snqi_mean | -0.1581 | -0.1100 | 0.0481 |
| orca | entering_elevator | 3 | 10 | time_to_goal_norm_mean | 0.9367 | 0.8830 | -0.0537 |
| orca | entering_elevator | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | entering_elevator | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_no_gesture | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| orca | intersection_no_gesture | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| orca | intersection_no_gesture | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_no_gesture | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_no_gesture | 3 | 10 | snqi_mean | -0.0311 | -0.0949 | -0.0638 |
| orca | intersection_no_gesture | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| orca | intersection_no_gesture | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | intersection_no_gesture | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_proceed | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| orca | intersection_proceed | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| orca | intersection_proceed | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_proceed | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_proceed | 3 | 10 | snqi_mean | -0.0311 | -0.0949 | -0.0638 |
| orca | intersection_proceed | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| orca | intersection_proceed | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | intersection_proceed | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_wait | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| orca | intersection_wait | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| orca | intersection_wait | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_wait | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | intersection_wait | 3 | 10 | snqi_mean | -0.0311 | -0.0949 | -0.0638 |
| orca | intersection_wait | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| orca | intersection_wait | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | intersection_wait | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_no_gesture | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| ppo | intersection_no_gesture | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| ppo | intersection_no_gesture | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_no_gesture | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_no_gesture | 3 | 10 | snqi_mean | -0.2399 | -0.2975 | -0.0576 |
| ppo | intersection_no_gesture | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| ppo | intersection_no_gesture | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | intersection_no_gesture | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_proceed | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| ppo | intersection_proceed | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| ppo | intersection_proceed | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_proceed | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_proceed | 3 | 10 | snqi_mean | -0.2392 | -0.3041 | -0.0649 |
| ppo | intersection_proceed | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| ppo | intersection_proceed | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | intersection_proceed | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_wait | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| ppo | intersection_wait | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| ppo | intersection_wait | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_wait | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | intersection_wait | 3 | 10 | snqi_mean | -0.2392 | -0.3041 | -0.0649 |
| ppo | intersection_wait | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| ppo | intersection_wait | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | intersection_wait | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_no_gesture | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| socnav_sampling | intersection_no_gesture | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| socnav_sampling | intersection_no_gesture | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_no_gesture | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_no_gesture | 3 | 10 | snqi_mean | -0.0311 | -0.0949 | -0.0638 |
| socnav_sampling | intersection_no_gesture | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| socnav_sampling | intersection_no_gesture | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | intersection_no_gesture | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_proceed | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| socnav_sampling | intersection_proceed | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| socnav_sampling | intersection_proceed | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_proceed | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_proceed | 3 | 10 | snqi_mean | -0.0311 | -0.0949 | -0.0638 |
| socnav_sampling | intersection_proceed | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| socnav_sampling | intersection_proceed | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | intersection_proceed | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_wait | 3 | 10 | success_mean | 0.3333 | 0.0000 | -0.3333 |
| socnav_sampling | intersection_wait | 3 | 10 | unfinished_mean | 0.6667 | 1.0000 | 0.3333 |
| socnav_sampling | intersection_wait | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_wait | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | intersection_wait | 3 | 10 | snqi_mean | -0.0311 | -0.0949 | -0.0638 |
| socnav_sampling | intersection_wait | 3 | 10 | time_to_goal_norm_mean | 0.9967 | 1.0000 | 0.0033 |
| socnav_sampling | intersection_wait | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | intersection_wait | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | entering_room | 3 | 10 | success_mean | 0.6667 | 0.4000 | -0.2667 |
| prediction_planner | entering_room | 3 | 10 | unfinished_mean | 0.3333 | 0.6000 | 0.2667 |
| prediction_planner | entering_room | 3 | 10 | collisions_mean | 0.3333 | 0.3000 | -0.0333 |
| prediction_planner | entering_room | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | entering_room | 3 | 10 | snqi_mean | 0.0378 | -0.0490 | -0.0868 |
| prediction_planner | entering_room | 3 | 10 | time_to_goal_norm_mean | 0.9400 | 0.9840 | 0.0440 |
| prediction_planner | entering_room | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | entering_room | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | circular_crossing | 3 | 10 | success_mean | 0.6667 | 0.9000 | 0.2333 |
| orca | circular_crossing | 3 | 10 | unfinished_mean | 0.3333 | 0.1000 | -0.2333 |
| orca | circular_crossing | 3 | 10 | collisions_mean | 0.3333 | 0.1000 | -0.2333 |
| orca | circular_crossing | 3 | 10 | near_misses_mean | 1.0000 | 0.3000 | -0.7000 |
| orca | circular_crossing | 3 | 10 | snqi_mean | -0.0403 | 0.0385 | 0.0788 |
| orca | circular_crossing | 3 | 10 | time_to_goal_norm_mean | 0.7667 | 0.7160 | -0.0507 |
| orca | circular_crossing | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | circular_crossing | 3 | 10 | comfort_exposure_mean | 0.1975 | 0.0882 | -0.1093 |
| ppo | circular_crossing | 3 | 10 | success_mean | 0.6667 | 0.9000 | 0.2333 |
| ppo | circular_crossing | 3 | 10 | unfinished_mean | 0.3333 | 0.1000 | -0.2333 |
| ppo | circular_crossing | 3 | 10 | collisions_mean | 0.3333 | 0.1000 | -0.2333 |
| ppo | circular_crossing | 3 | 10 | near_misses_mean | 0.6667 | 0.3000 | -0.3667 |
| ppo | circular_crossing | 3 | 10 | snqi_mean | -0.1323 | -0.1229 | 0.0094 |
| ppo | circular_crossing | 3 | 10 | time_to_goal_norm_mean | 0.6967 | 0.6120 | -0.0847 |
| ppo | circular_crossing | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | circular_crossing | 3 | 10 | comfort_exposure_mean | 0.1964 | 0.0920 | -0.1044 |
| prediction_planner | circular_crossing | 3 | 10 | success_mean | 0.6667 | 0.9000 | 0.2333 |
| prediction_planner | circular_crossing | 3 | 10 | unfinished_mean | 0.3333 | 0.1000 | -0.2333 |
| prediction_planner | circular_crossing | 3 | 10 | collisions_mean | 0.3333 | 0.1000 | -0.2333 |
| prediction_planner | circular_crossing | 3 | 10 | near_misses_mean | 1.3333 | 0.4000 | -0.9333 |
| prediction_planner | circular_crossing | 3 | 10 | snqi_mean | -0.0217 | 0.0542 | 0.0759 |
| prediction_planner | circular_crossing | 3 | 10 | time_to_goal_norm_mean | 0.8267 | 0.7720 | -0.0547 |
| prediction_planner | circular_crossing | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | circular_crossing | 3 | 10 | comfort_exposure_mean | 0.1925 | 0.0864 | -0.1061 |
| socnav_sampling | circular_crossing | 3 | 10 | success_mean | 0.6667 | 0.9000 | 0.2333 |
| socnav_sampling | circular_crossing | 3 | 10 | unfinished_mean | 0.3333 | 0.1000 | -0.2333 |
| socnav_sampling | circular_crossing | 3 | 10 | collisions_mean | 0.3333 | 0.1000 | -0.2333 |
| socnav_sampling | circular_crossing | 3 | 10 | near_misses_mean | 0.6667 | 0.3000 | -0.3667 |
| socnav_sampling | circular_crossing | 3 | 10 | snqi_mean | -0.0222 | 0.0447 | 0.0669 |
| socnav_sampling | circular_crossing | 3 | 10 | time_to_goal_norm_mean | 0.7367 | 0.6240 | -0.1127 |
| socnav_sampling | circular_crossing | 3 | 10 | path_efficiency_mean | 0.9907 | 1.0000 | 0.0093 |
| socnav_sampling | circular_crossing | 3 | 10 | comfort_exposure_mean | 0.1948 | 0.0912 | -0.1036 |
| ppo | doorway | 9 | 30 | success_mean | 0.4444 | 0.2333 | -0.2111 |
| ppo | doorway | 9 | 30 | unfinished_mean | 0.5556 | 0.7667 | 0.2111 |
| ppo | doorway | 9 | 30 | collisions_mean | 0.2222 | 0.3667 | 0.1445 |
| ppo | doorway | 9 | 30 | near_misses_mean | 11.8889 | 8.8000 | -3.0889 |
| ppo | doorway | 9 | 30 | snqi_mean | -0.8936 | -0.8724 | 0.0212 |
| ppo | doorway | 9 | 30 | time_to_goal_norm_mean | 0.9378 | 0.9317 | -0.0061 |
| ppo | doorway | 9 | 30 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | doorway | 9 | 30 | comfort_exposure_mean | 0.0210 | 0.0165 | -0.0045 |
| orca | robot_crowding | 3 | 10 | success_mean | 0.0000 | 0.2000 | 0.2000 |
| orca | robot_crowding | 3 | 10 | unfinished_mean | 1.0000 | 0.8000 | -0.2000 |
| orca | robot_crowding | 3 | 10 | collisions_mean | 0.3333 | 0.4000 | 0.0667 |
| orca | robot_crowding | 3 | 10 | near_misses_mean | 29.0000 | 20.5000 | -8.5000 |
| orca | robot_crowding | 3 | 10 | snqi_mean | -1.2204 | -0.8207 | 0.3997 |
| orca | robot_crowding | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9770 | -0.0230 |
| orca | robot_crowding | 3 | 10 | path_efficiency_mean | 1.0000 | 0.9817 | -0.0183 |
| orca | robot_crowding | 3 | 10 | comfort_exposure_mean | 0.1807 | 0.2142 | 0.0335 |
| prediction_planner | group_crossing | 9 | 30 | success_mean | 0.1111 | 0.0000 | -0.1111 |
| prediction_planner | group_crossing | 9 | 30 | unfinished_mean | 0.8889 | 1.0000 | 0.1111 |
| prediction_planner | group_crossing | 9 | 30 | collisions_mean | 0.1111 | 0.1000 | -0.0111 |
| prediction_planner | group_crossing | 9 | 30 | near_misses_mean | 10.3333 | 12.7000 | 2.3667 |
| prediction_planner | group_crossing | 9 | 30 | snqi_mean | -0.2032 | -0.2474 | -0.0442 |
| prediction_planner | group_crossing | 9 | 30 | time_to_goal_norm_mean | 0.9989 | 1.0000 | 0.0011 |
| prediction_planner | group_crossing | 9 | 30 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | group_crossing | 9 | 30 | comfort_exposure_mean | 0.0149 | 0.0217 | 0.0068 |
| prediction_planner | join_group | 3 | 10 | success_mean | 0.0000 | 0.1000 | 0.1000 |
| prediction_planner | join_group | 3 | 10 | unfinished_mean | 1.0000 | 0.9000 | -0.1000 |
| prediction_planner | join_group | 3 | 10 | collisions_mean | 0.3333 | 0.4000 | 0.0667 |
| prediction_planner | join_group | 3 | 10 | near_misses_mean | 25.6667 | 23.4000 | -2.2667 |
| prediction_planner | join_group | 3 | 10 | snqi_mean | -0.3726 | -0.3866 | -0.0140 |
| prediction_planner | join_group | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9950 | -0.0050 |
| prediction_planner | join_group | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | join_group | 3 | 10 | comfort_exposure_mean | 0.0655 | 0.0570 | -0.0085 |
| prediction_planner | leave_group | 3 | 10 | success_mean | 0.0000 | 0.1000 | 0.1000 |
| prediction_planner | leave_group | 3 | 10 | unfinished_mean | 1.0000 | 0.9000 | -0.1000 |
| prediction_planner | leave_group | 3 | 10 | collisions_mean | 0.3333 | 0.6000 | 0.2667 |
| prediction_planner | leave_group | 3 | 10 | near_misses_mean | 48.3333 | 37.1000 | -11.2333 |
| prediction_planner | leave_group | 3 | 10 | snqi_mean | -0.4519 | -0.4802 | -0.0283 |
| prediction_planner | leave_group | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 0.9920 | -0.0080 |
| prediction_planner | leave_group | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | leave_group | 3 | 10 | comfort_exposure_mean | 0.1190 | 0.0950 | -0.0240 |
| ppo | exiting_elevator | 3 | 10 | success_mean | 1.0000 | 0.9000 | -0.1000 |
| ppo | exiting_elevator | 3 | 10 | unfinished_mean | 0.0000 | 0.1000 | 0.1000 |
| ppo | exiting_elevator | 3 | 10 | collisions_mean | 0.0000 | 0.1000 | 0.1000 |
| ppo | exiting_elevator | 3 | 10 | near_misses_mean | 11.0000 | 10.5000 | -0.5000 |
| ppo | exiting_elevator | 3 | 10 | snqi_mean | -0.2959 | -0.3286 | -0.0327 |
| ppo | exiting_elevator | 3 | 10 | time_to_goal_norm_mean | 0.5467 | 0.5690 | 0.0223 |
| ppo | exiting_elevator | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | exiting_elevator | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | doorway | 9 | 30 | success_mean | 0.2222 | 0.1333 | -0.0889 |
| prediction_planner | doorway | 9 | 30 | unfinished_mean | 0.7778 | 0.8667 | 0.0889 |
| prediction_planner | doorway | 9 | 30 | collisions_mean | 0.2222 | 0.3333 | 0.1111 |
| prediction_planner | doorway | 9 | 30 | near_misses_mean | 24.4444 | 25.7667 | 1.3223 |
| prediction_planner | doorway | 9 | 30 | snqi_mean | -0.4146 | -0.4540 | -0.0394 |
| prediction_planner | doorway | 9 | 30 | time_to_goal_norm_mean | 0.9756 | 0.9823 | 0.0067 |
| prediction_planner | doorway | 9 | 30 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | doorway | 9 | 30 | comfort_exposure_mean | 0.0193 | 0.0185 | -0.0008 |
| orca | group_crossing | 9 | 30 | success_mean | 0.8889 | 0.8000 | -0.0889 |
| orca | group_crossing | 9 | 30 | unfinished_mean | 0.1111 | 0.2000 | 0.0889 |
| orca | group_crossing | 9 | 30 | collisions_mean | 0.0000 | 0.0333 | 0.0333 |
| orca | group_crossing | 9 | 30 | near_misses_mean | 4.7778 | 5.4667 | 0.6889 |
| orca | group_crossing | 9 | 30 | snqi_mean | -0.0103 | -0.0843 | -0.0740 |
| orca | group_crossing | 9 | 30 | time_to_goal_norm_mean | 0.8800 | 0.9190 | 0.0390 |
| orca | group_crossing | 9 | 30 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | group_crossing | 9 | 30 | comfort_exposure_mean | 0.0194 | 0.0274 | 0.0080 |
| prediction_planner | exiting_elevator | 3 | 10 | success_mean | 0.3333 | 0.4000 | 0.0667 |
| prediction_planner | exiting_elevator | 3 | 10 | unfinished_mean | 0.6667 | 0.6000 | -0.0667 |
| prediction_planner | exiting_elevator | 3 | 10 | collisions_mean | 0.3333 | 0.3000 | -0.0333 |
| prediction_planner | exiting_elevator | 3 | 10 | near_misses_mean | 27.0000 | 24.5000 | -2.5000 |
| prediction_planner | exiting_elevator | 3 | 10 | snqi_mean | -0.2445 | -0.2630 | -0.0185 |
| prediction_planner | exiting_elevator | 3 | 10 | time_to_goal_norm_mean | 0.9733 | 0.9380 | -0.0353 |
| prediction_planner | exiting_elevator | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | exiting_elevator | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | group_crossing | 9 | 30 | success_mean | 0.5556 | 0.5000 | -0.0556 |
| socnav_sampling | group_crossing | 9 | 30 | unfinished_mean | 0.4444 | 0.5000 | 0.0556 |
| socnav_sampling | group_crossing | 9 | 30 | collisions_mean | 0.4444 | 0.5000 | 0.0556 |
| socnav_sampling | group_crossing | 9 | 30 | near_misses_mean | 0.7778 | 2.4333 | 1.6555 |
| socnav_sampling | group_crossing | 9 | 30 | snqi_mean | -0.0231 | -0.1297 | -0.1066 |
| socnav_sampling | group_crossing | 9 | 30 | time_to_goal_norm_mean | 0.9167 | 0.9343 | 0.0176 |
| socnav_sampling | group_crossing | 9 | 30 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | group_crossing | 9 | 30 | comfort_exposure_mean | 0.0234 | 0.0263 | 0.0029 |
| ppo | bottleneck | 12 | 40 | success_mean | 0.0000 | 0.0500 | 0.0500 |
| ppo | bottleneck | 12 | 40 | unfinished_mean | 1.0000 | 0.9500 | -0.0500 |
| ppo | bottleneck | 12 | 40 | collisions_mean | 0.1667 | 0.2500 | 0.0833 |
| ppo | bottleneck | 12 | 40 | near_misses_mean | 6.3333 | 6.0250 | -0.3083 |
| ppo | bottleneck | 12 | 40 | snqi_mean | -0.3832 | -0.3869 | -0.0037 |
| ppo | bottleneck | 12 | 40 | time_to_goal_norm_mean | 1.0000 | 0.9982 | -0.0018 |
| ppo | bottleneck | 12 | 40 | path_efficiency_mean | 1.0000 | 0.9951 | -0.0049 |
| ppo | bottleneck | 12 | 40 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| socnav_sampling | bottleneck | 12 | 40 | success_mean | 0.0000 | 0.0500 | 0.0500 |
| socnav_sampling | bottleneck | 12 | 40 | unfinished_mean | 1.0000 | 0.9500 | -0.0500 |
| socnav_sampling | bottleneck | 12 | 40 | collisions_mean | 0.6667 | 0.6500 | -0.0167 |
| socnav_sampling | bottleneck | 12 | 40 | near_misses_mean | 6.5000 | 4.8000 | -1.7000 |
| socnav_sampling | bottleneck | 12 | 40 | snqi_mean | -0.2726 | -0.3143 | -0.0417 |
| socnav_sampling | bottleneck | 12 | 40 | time_to_goal_norm_mean | 1.0000 | 0.9992 | -0.0008 |
| socnav_sampling | bottleneck | 12 | 40 | path_efficiency_mean | 0.9597 | 0.9764 | 0.0167 |
| socnav_sampling | bottleneck | 12 | 40 | comfort_exposure_mean | 0.0076 | 0.0047 | -0.0029 |
| socnav_sampling | doorway | 9 | 30 | success_mean | 0.0000 | 0.0333 | 0.0333 |
| socnav_sampling | doorway | 9 | 30 | unfinished_mean | 1.0000 | 0.9667 | -0.0333 |
| socnav_sampling | doorway | 9 | 30 | collisions_mean | 1.0000 | 0.9667 | -0.0333 |
| socnav_sampling | doorway | 9 | 30 | near_misses_mean | 1.0000 | 1.8000 | 0.8000 |
| socnav_sampling | doorway | 9 | 30 | snqi_mean | -0.1956 | -0.3158 | -0.1202 |
| socnav_sampling | doorway | 9 | 30 | time_to_goal_norm_mean | 1.0000 | 0.9830 | -0.0170 |
| socnav_sampling | doorway | 9 | 30 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| socnav_sampling | doorway | 9 | 30 | comfort_exposure_mean | 0.0263 | 0.0272 | 0.0009 |
| orca | leave_group | 3 | 10 | success_mean | 0.6667 | 0.7000 | 0.0333 |
| orca | leave_group | 3 | 10 | unfinished_mean | 0.3333 | 0.3000 | -0.0333 |
| orca | leave_group | 3 | 10 | collisions_mean | 0.3333 | 0.3000 | -0.0333 |
| orca | leave_group | 3 | 10 | near_misses_mean | 21.3333 | 23.9000 | 2.5667 |
| orca | leave_group | 3 | 10 | snqi_mean | -0.2678 | -0.2830 | -0.0152 |
| orca | leave_group | 3 | 10 | time_to_goal_norm_mean | 0.7533 | 0.7910 | 0.0377 |
| orca | leave_group | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | leave_group | 3 | 10 | comfort_exposure_mean | 0.0847 | 0.0866 | 0.0019 |
| ppo | robot_crowding | 3 | 10 | success_mean | 0.6667 | 0.7000 | 0.0333 |
| ppo | robot_crowding | 3 | 10 | unfinished_mean | 0.3333 | 0.3000 | -0.0333 |
| ppo | robot_crowding | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | robot_crowding | 3 | 10 | near_misses_mean | 0.0000 | 7.0000 | 7.0000 |
| ppo | robot_crowding | 3 | 10 | snqi_mean | -0.3745 | -0.5844 | -0.2099 |
| ppo | robot_crowding | 3 | 10 | time_to_goal_norm_mean | 0.8967 | 0.8970 | 0.0003 |
| ppo | robot_crowding | 3 | 10 | path_efficiency_mean | 0.8717 | 0.8359 | -0.0358 |
| ppo | robot_crowding | 3 | 10 | comfort_exposure_mean | 0.1516 | 0.1801 | 0.0285 |
| prediction_planner | entering_elevator | 3 | 10 | success_mean | 0.6667 | 0.7000 | 0.0333 |
| prediction_planner | entering_elevator | 3 | 10 | unfinished_mean | 0.3333 | 0.3000 | -0.0333 |
| prediction_planner | entering_elevator | 3 | 10 | collisions_mean | 0.3333 | 0.3000 | -0.0333 |
| prediction_planner | entering_elevator | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| prediction_planner | entering_elevator | 3 | 10 | snqi_mean | 0.0605 | 0.0307 | -0.0298 |
| prediction_planner | entering_elevator | 3 | 10 | time_to_goal_norm_mean | 0.7000 | 0.7440 | 0.0440 |
| prediction_planner | entering_elevator | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| prediction_planner | entering_elevator | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | bottleneck | 12 | 40 | success_mean | 0.0000 | 0.0250 | 0.0250 |
| orca | bottleneck | 12 | 40 | unfinished_mean | 1.0000 | 0.9750 | -0.0250 |
| orca | bottleneck | 12 | 40 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| orca | bottleneck | 12 | 40 | near_misses_mean | 4.4167 | 5.0250 | 0.6083 |
| orca | bottleneck | 12 | 40 | snqi_mean | -0.1911 | -0.2108 | -0.0197 |
| orca | bottleneck | 12 | 40 | time_to_goal_norm_mean | 1.0000 | 0.9995 | -0.0005 |
| orca | bottleneck | 12 | 40 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | bottleneck | 12 | 40 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| ppo | group_crossing | 9 | 30 | success_mean | 0.7778 | 0.8000 | 0.0222 |
| ppo | group_crossing | 9 | 30 | unfinished_mean | 0.2222 | 0.2000 | -0.0222 |
| ppo | group_crossing | 9 | 30 | collisions_mean | 0.0000 | 0.0333 | 0.0333 |
| ppo | group_crossing | 9 | 30 | near_misses_mean | 1.1111 | 3.7000 | 2.5889 |
| ppo | group_crossing | 9 | 30 | snqi_mean | -0.1560 | -0.2515 | -0.0955 |
| ppo | group_crossing | 9 | 30 | time_to_goal_norm_mean | 0.8689 | 0.9160 | 0.0471 |
| ppo | group_crossing | 9 | 30 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| ppo | group_crossing | 9 | 30 | comfort_exposure_mean | 0.0179 | 0.0194 | 0.0015 |
| orca | doorway | 9 | 30 | success_mean | 0.1111 | 0.1000 | -0.0111 |
| orca | doorway | 9 | 30 | unfinished_mean | 0.8889 | 0.9000 | 0.0111 |
| orca | doorway | 9 | 30 | collisions_mean | 0.1111 | 0.0333 | -0.0778 |
| orca | doorway | 9 | 30 | near_misses_mean | 13.1111 | 14.2000 | 1.0889 |
| orca | doorway | 9 | 30 | snqi_mean | -0.7528 | -0.7547 | -0.0019 |
| orca | doorway | 9 | 30 | time_to_goal_norm_mean | 0.9867 | 0.9840 | -0.0027 |
| orca | doorway | 9 | 30 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| orca | doorway | 9 | 30 | comfort_exposure_mean | 0.0123 | 0.0196 | 0.0073 |
| goal | accompanying_peer | 3 | 10 | success_mean | 0.0000 | 0.0000 | 0.0000 |
| goal | accompanying_peer | 3 | 10 | unfinished_mean | 1.0000 | 1.0000 | 0.0000 |
| goal | accompanying_peer | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| goal | accompanying_peer | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| goal | accompanying_peer | 3 | 10 | snqi_mean | -0.1023 | -0.1048 | -0.0025 |
| goal | accompanying_peer | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 1.0000 | 0.0000 |
| goal | accompanying_peer | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| goal | accompanying_peer | 3 | 10 | comfort_exposure_mean | 0.0100 | 0.0100 | 0.0000 |
| goal | blind_corner | 3 | 10 | success_mean | 0.0000 | 0.0000 | 0.0000 |
| goal | blind_corner | 3 | 10 | unfinished_mean | 1.0000 | 1.0000 | 0.0000 |
| goal | blind_corner | 3 | 10 | collisions_mean | 0.0000 | 0.0000 | 0.0000 |
| goal | blind_corner | 3 | 10 | near_misses_mean | 0.0000 | 0.0000 | 0.0000 |
| goal | blind_corner | 3 | 10 | snqi_mean | -0.1111 | -0.1227 | -0.0116 |
| goal | blind_corner | 3 | 10 | time_to_goal_norm_mean | 1.0000 | 1.0000 | 0.0000 |
| goal | blind_corner | 3 | 10 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| goal | blind_corner | 3 | 10 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
| goal | bottleneck | 12 | 40 | success_mean | 0.0000 | 0.0000 | 0.0000 |
| goal | bottleneck | 12 | 40 | unfinished_mean | 1.0000 | 1.0000 | 0.0000 |
| goal | bottleneck | 12 | 40 | collisions_mean | 0.0000 | 0.0750 | 0.0750 |
| goal | bottleneck | 12 | 40 | near_misses_mean | 0.2500 | 0.0750 | -0.1750 |
| goal | bottleneck | 12 | 40 | snqi_mean | -0.0986 | -0.1095 | -0.0109 |
| goal | bottleneck | 12 | 40 | time_to_goal_norm_mean | 1.0000 | 1.0000 | 0.0000 |
| goal | bottleneck | 12 | 40 | path_efficiency_mean | 1.0000 | 1.0000 | 0.0000 |
| goal | bottleneck | 12 | 40 | comfort_exposure_mean | 0.0000 | 0.0000 | 0.0000 |
