# Latest Policy Sweep Comparison (2026-01-28)

Runs compared: fast_pysf_planner (oracle/GT), PPO (learned), SocNav ORCA (heuristic).

## Runs
- `fast_pysf_planner` (oracle/GT): `output/benchmarks/policy_analysis_fast_pysf_planner_20260128_132624`
- `ppo` (learned): `output/benchmarks/policy_analysis_ppo_20260128_132624`
- `socnav_orca` (heuristic): `output/benchmarks/policy_analysis_socnav_orca_20260128_132624`

## Aggregate Metrics (mean over episodes)
| Run | Success | Collision | TTG_norm | AvgSpeed | NearMiss | Comfort | Jerk | Curvature | Jerk>=0.1 | Curv>=0.1 | Energy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fast_pysf_planner | 0.795 | 0.167 | 0.244 | 0.574 | 3.447 | 0.014 | 0.009 | 47.530 | 0.012 | 0.231 | 23.573 |
| ppo | 0.949 | 0.037 | 0.077 | 1.440 | 0.228 | 0.011 | 0.572 | 492.410 | 0.572 | 0.447 | 179.228 |
| socnav_orca | 0.963 | 0.014 | 0.067 | 1.349 | 0.809 | 0.017 | 0.065 | 0.283 | 0.066 | 0.081 | 27.107 |

## Worst Collision Scenarios (top 5 per run)
### fast_pysf_planner
- classic_overtaking_low: collisions_mean=8.400 success_mean=0.000
- classic_crossing_high: collisions_mean=7.000 success_mean=0.200
- classic_crossing_medium: collisions_mean=5.200 success_mean=0.400
- francis2023_crowd_navigation: collisions_mean=5.000 success_mean=0.400
- classic_overtaking_medium: collisions_mean=4.200 success_mean=0.600

### ppo
- francis2023_leave_group: collisions_mean=2.000 success_mean=0.200
- classic_overtaking_low: collisions_mean=1.000 success_mean=0.600
- classic_overtaking_medium: collisions_mean=0.800 success_mean=0.600
- classic_crossing_low: collisions_mean=0.000 success_mean=1.000
- classic_crossing_medium: collisions_mean=0.000 success_mean=1.000

### socnav_orca
- francis2023_robot_crowding: collisions_mean=2.000 success_mean=0.600
- francis2023_leave_group: collisions_mean=0.600 success_mean=0.800
- classic_crossing_low: collisions_mean=0.000 success_mean=1.000
- classic_crossing_medium: collisions_mean=0.000 success_mean=1.000
- classic_crossing_high: collisions_mean=0.000 success_mean=1.000

## Lowest Success Scenarios (bottom 5 per run)
### fast_pysf_planner
- classic_overtaking_low: success_mean=0.000
- francis2023_exiting_elevator: success_mean=0.000
- francis2023_following_human: success_mean=0.000
- francis2023_leading_human: success_mean=0.000
- classic_crossing_high: success_mean=0.200

### ppo
- francis2023_leave_group: success_mean=0.200
- francis2023_circular_crossing: success_mean=0.400
- classic_overtaking_low: success_mean=0.600
- classic_overtaking_medium: success_mean=0.600
- classic_crossing_low: success_mean=1.000

### socnav_orca
- francis2023_circular_crossing: success_mean=0.400
- classic_overtaking_medium: success_mean=0.600
- francis2023_robot_crowding: success_mean=0.600
- francis2023_leave_group: success_mean=0.800
- classic_crossing_low: success_mean=1.000

## Top Problem Episodes (score = collisions*10 + comfort*2 + failure)
### fast_pysf_planner
- classic_overtaking_medium seed=123 score=150.00 coll=15.0 comfort=0.000 status=collision video=`output/recordings/policy_analysis_fast_pysf_planner_20260128_132624/classic_overtaking_medium_seed123_fast_pysf_planner.mp4`
- classic_crossing_high seed=231 score=140.00 coll=14.0 comfort=0.002 status=collision video=`output/recordings/policy_analysis_fast_pysf_planner_20260128_132624/classic_crossing_high_seed231_fast_pysf_planner.mp4`
- francis2023_crowd_navigation seed=992 score=120.09 coll=12.0 comfort=0.044 status=collision video=`output/recordings/policy_analysis_fast_pysf_planner_20260128_132624/francis2023_crowd_navigation_seed992_fast_pysf_planner.mp4`
- classic_crossing_medium seed=1337 score=100.00 coll=10.0 comfort=0.001 status=collision video=`output/recordings/policy_analysis_fast_pysf_planner_20260128_132624/classic_crossing_medium_seed1337_fast_pysf_planner.mp4`
- classic_crossing_low seed=1337 score=100.00 coll=10.0 comfort=0.000 status=collision video=`output/recordings/policy_analysis_fast_pysf_planner_20260128_132624/classic_crossing_low_seed1337_fast_pysf_planner.mp4`

### ppo
- francis2023_leave_group seed=777 score=40.00 coll=4.0 comfort=0.000 status=collision video=`output/recordings/policy_analysis_ppo_20260128_132624/francis2023_leave_group_seed777_ppo.mp4`
- classic_overtaking_medium seed=231 score=30.01 coll=3.0 comfort=0.004 status=collision video=`output/recordings/policy_analysis_ppo_20260128_132624/classic_overtaking_medium_seed231_ppo.mp4`
- classic_overtaking_low seed=992 score=30.00 coll=3.0 comfort=0.000 status=collision video=`output/recordings/policy_analysis_ppo_20260128_132624/classic_overtaking_low_seed992_ppo.mp4`
- francis2023_leave_group seed=992 score=30.00 coll=3.0 comfort=0.000 status=collision video=`output/recordings/policy_analysis_ppo_20260128_132624/francis2023_leave_group_seed992_ppo.mp4`
- classic_overtaking_low seed=777 score=20.00 coll=2.0 comfort=0.000 status=collision video=`output/recordings/policy_analysis_ppo_20260128_132624/classic_overtaking_low_seed777_ppo.mp4`

### socnav_orca
- francis2023_robot_crowding seed=1337 score=90.06 coll=9.0 comfort=0.030 status=collision video=`output/recordings/policy_analysis_socnav_orca_20260128_132624/francis2023_robot_crowding_seed1337_socnav_orca.mp4`
- francis2023_leave_group seed=231 score=30.28 coll=3.0 comfort=0.138 status=collision video=`output/recordings/policy_analysis_socnav_orca_20260128_132624/francis2023_leave_group_seed231_socnav_orca.mp4`
- francis2023_robot_crowding seed=777 score=10.22 coll=1.0 comfort=0.111 status=collision video=`output/recordings/policy_analysis_socnav_orca_20260128_132624/francis2023_robot_crowding_seed777_socnav_orca.mp4`
- francis2023_circular_crossing seed=1337 score=2.00 coll=0.0 comfort=0.500 status=failure video=`output/recordings/policy_analysis_socnav_orca_20260128_132624/francis2023_circular_crossing_seed1337_socnav_orca.mp4`
- francis2023_circular_crossing seed=992 score=1.67 coll=0.0 comfort=0.333 status=failure video=`output/recordings/policy_analysis_socnav_orca_20260128_132624/francis2023_circular_crossing_seed992_socnav_orca.mp4`

## Diagnostics

### Check 1: shortest_path_len presence
- `fast_pysf_planner`: shortest_path_present_frac=1.000
- `ppo`: shortest_path_present_frac=1.000
- `socnav_orca`: shortest_path_present_frac=1.000

### Check 2: path_efficiency saturation
- `fast_pysf_planner`: path_efficiency_ge_0.999_frac=0.981
- `ppo`: path_efficiency_ge_0.999_frac=1.000
- `socnav_orca`: path_efficiency_ge_0.999_frac=1.000

### Check 3: low-speed filter (ε=0.1 m/s)
Reason: curvature divides by |v|^3 and jerk uses finite differences; near-zero speeds inflate noise.
- `fast_pysf_planner`: low_speed_frac=0.033
- `ppo`: low_speed_frac=0.014
- `socnav_orca`: low_speed_frac=0.014

## Frame Snapshots
Sample frames extracted at 25/50/75% for the top 3 problem episodes per run.

### fast_pysf_planner
- `output/analysis/latest_policy_sweep_20260128_132624/frames/fast_pysf_planner/classic_crossing_high_seed231_fast_pysf_planner_t25.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/fast_pysf_planner/classic_crossing_high_seed231_fast_pysf_planner_t50.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/fast_pysf_planner/classic_crossing_high_seed231_fast_pysf_planner_t75.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/fast_pysf_planner/classic_overtaking_medium_seed123_fast_pysf_planner_t25.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/fast_pysf_planner/classic_overtaking_medium_seed123_fast_pysf_planner_t50.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/fast_pysf_planner/classic_overtaking_medium_seed123_fast_pysf_planner_t75.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/fast_pysf_planner/francis2023_crowd_navigation_seed992_fast_pysf_planner_t25.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/fast_pysf_planner/francis2023_crowd_navigation_seed992_fast_pysf_planner_t50.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/fast_pysf_planner/francis2023_crowd_navigation_seed992_fast_pysf_planner_t75.png`

### ppo
- `output/analysis/latest_policy_sweep_20260128_132624/frames/ppo/classic_overtaking_low_seed992_ppo_t25.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/ppo/classic_overtaking_low_seed992_ppo_t50.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/ppo/classic_overtaking_low_seed992_ppo_t75.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/ppo/classic_overtaking_medium_seed231_ppo_t25.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/ppo/classic_overtaking_medium_seed231_ppo_t50.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/ppo/classic_overtaking_medium_seed231_ppo_t75.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/ppo/francis2023_leave_group_seed777_ppo_t25.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/ppo/francis2023_leave_group_seed777_ppo_t50.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/ppo/francis2023_leave_group_seed777_ppo_t75.png`

### socnav_orca
- `output/analysis/latest_policy_sweep_20260128_132624/frames/socnav_orca/francis2023_leave_group_seed231_socnav_orca_t25.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/socnav_orca/francis2023_leave_group_seed231_socnav_orca_t50.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/socnav_orca/francis2023_leave_group_seed231_socnav_orca_t75.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/socnav_orca/francis2023_robot_crowding_seed1337_socnav_orca_t25.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/socnav_orca/francis2023_robot_crowding_seed1337_socnav_orca_t50.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/socnav_orca/francis2023_robot_crowding_seed1337_socnav_orca_t75.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/socnav_orca/francis2023_robot_crowding_seed777_socnav_orca_t25.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/socnav_orca/francis2023_robot_crowding_seed777_socnav_orca_t50.png`
- `output/analysis/latest_policy_sweep_20260128_132624/frames/socnav_orca/francis2023_robot_crowding_seed777_socnav_orca_t75.png`
