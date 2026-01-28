# Planner Comparison (2026-01-28)

Comparing policy analysis sweeps on `classic_interactions_francis2023.yaml`.

## Runs
- `fast_pysf_planner`: `output/benchmarks/policy_analysis_fast_pysf_planner_20260127_191536`
- `ppo`: `output/benchmarks/policy_analysis_ppo_20260128_094234`
- `socnav_orca`: `output/benchmarks/policy_analysis_socnav_orca_20260128_104400`

## Aggregate Metrics (mean over episodes)
| Run | Success | Collision | TTG_norm | AvgSpeed | NearMiss | Comfort | PedF95 | Jerk | Energy |
|---|---|---|---|---|---|---|---|---|---|
| fast_pysf_planner | 0.795 | 0.167 | 0.244 | 0.574 | 3.447 | 0.014 | 0.900 | 0.009 | 23.573 |
| ppo | 0.949 | 0.037 | 0.077 | 1.440 | 0.228 | 0.011 | 0.884 | 0.572 | 179.228 |
| socnav_orca | 0.963 | 0.014 | 0.067 | 1.349 | 0.809 | 0.017 | 1.014 | 0.065 | 27.107 |

## Highlights
- `socnav_orca` has the best success/collision balance (highest success, lowest collision rate).
- `ppo` is fastest in practice (high avg speed) but shows much higher jerk/energy (aggressive control).
- `fast_pysf_planner` is smooth/low-energy but has substantially higher collision and near-miss rates.

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
- classic_overtaking_medium seed=123 score=150.00 coll=15.0 comfort=0.000 status=collision video=`output/recordings/policy_analysis_fast_pysf_planner_20260127_191536/classic_overtaking_medium_seed123_fast_pysf_planner.mp4`
- classic_crossing_high seed=231 score=140.00 coll=14.0 comfort=0.002 status=collision video=`output/recordings/policy_analysis_fast_pysf_planner_20260127_191536/classic_crossing_high_seed231_fast_pysf_planner.mp4`
- francis2023_crowd_navigation seed=992 score=120.09 coll=12.0 comfort=0.044 status=collision video=`output/recordings/policy_analysis_fast_pysf_planner_20260127_191536/francis2023_crowd_navigation_seed992_fast_pysf_planner.mp4`
- classic_crossing_medium seed=1337 score=100.00 coll=10.0 comfort=0.001 status=collision video=`output/recordings/policy_analysis_fast_pysf_planner_20260127_191536/classic_crossing_medium_seed1337_fast_pysf_planner.mp4`
- classic_crossing_low seed=1337 score=100.00 coll=10.0 comfort=0.000 status=collision video=`output/recordings/policy_analysis_fast_pysf_planner_20260127_191536/classic_crossing_low_seed1337_fast_pysf_planner.mp4`

### ppo
- francis2023_leave_group seed=777 score=40.00 coll=4.0 comfort=0.000 status=collision video=`output/recordings/policy_analysis_ppo_20260128_094234/francis2023_leave_group_seed777_ppo.mp4`
- classic_overtaking_medium seed=231 score=30.01 coll=3.0 comfort=0.004 status=collision video=`output/recordings/policy_analysis_ppo_20260128_094234/classic_overtaking_medium_seed231_ppo.mp4`
- classic_overtaking_low seed=992 score=30.00 coll=3.0 comfort=0.000 status=collision video=`output/recordings/policy_analysis_ppo_20260128_094234/classic_overtaking_low_seed992_ppo.mp4`
- francis2023_leave_group seed=992 score=30.00 coll=3.0 comfort=0.000 status=collision video=`output/recordings/policy_analysis_ppo_20260128_094234/francis2023_leave_group_seed992_ppo.mp4`
- classic_overtaking_low seed=777 score=20.00 coll=2.0 comfort=0.000 status=collision video=`output/recordings/policy_analysis_ppo_20260128_094234/classic_overtaking_low_seed777_ppo.mp4`

### socnav_orca
- francis2023_robot_crowding seed=1337 score=90.06 coll=9.0 comfort=0.030 status=collision video=`output/recordings/policy_analysis_socnav_orca_20260128_104400/francis2023_robot_crowding_seed1337_socnav_orca.mp4`
- francis2023_leave_group seed=231 score=30.28 coll=3.0 comfort=0.138 status=collision video=`output/recordings/policy_analysis_socnav_orca_20260128_104400/francis2023_leave_group_seed231_socnav_orca.mp4`
- francis2023_robot_crowding seed=777 score=10.22 coll=1.0 comfort=0.111 status=collision video=`output/recordings/policy_analysis_socnav_orca_20260128_104400/francis2023_robot_crowding_seed777_socnav_orca.mp4`
- francis2023_circular_crossing seed=1337 score=2.00 coll=0.0 comfort=0.500 status=failure video=`output/recordings/policy_analysis_socnav_orca_20260128_104400/francis2023_circular_crossing_seed1337_socnav_orca.mp4`
- francis2023_circular_crossing seed=992 score=1.67 coll=0.0 comfort=0.333 status=failure video=`output/recordings/policy_analysis_socnav_orca_20260128_104400/francis2023_circular_crossing_seed992_socnav_orca.mp4`

## Notes
- Video files exist for all episodes in each run; paths above follow the naming convention used by the analysis script.
- Frame-level analysis was not performed here; we can extract frames for specific episodes on request.

## Diagnostics

Sanity checks for path efficiency saturation, shortest-path proxy, and low-speed curvature/jerk behavior.

### Check 1: Shortest-path validity (proxy)
- `shortest_path_len` is not stored in `episodes.jsonl`, so we cannot directly count NaN/0 values.
- Proxy: fraction of `path_efficiency` that is NaN (should be 0 if shortest path was valid).
- `fast_pysf_planner`: path_efficiency_nan_frac=0.000
- `ppo`: path_efficiency_nan_frac=0.000
- `socnav_orca`: path_efficiency_nan_frac=0.000

### Check 2: Path-efficiency saturation/clamping
Fraction of episodes with `path_efficiency >= 0.999` (values are clipped to 1.0).
- `fast_pysf_planner`: ge_0.999_frac=0.981
- `ppo`: ge_0.999_frac=1.000
- `socnav_orca`: ge_0.999_frac=1.000

Interpretation: PPO and ORCA are fully saturated at 1.0; fast_pysf is ~98% saturated. This suggests either near-optimal paths or that the shortest-path estimate is systematically >= actual path length (clamping).

### Check 3: Low-speed curvature/jerk sensitivity
We flag how often `avg_speed < 0.2` and report curvature/jerk stats there.
- `fast_pysf_planner`: low_speed_frac=0.149
  - low_speed_curvature mean=279.875 median=49.726 p90=857.732
  - low_speed_jerk mean=0.006 median=0.004 p90=0.008
  - corr(avg_speed, curvature_mean)=-0.526 corr(avg_speed, jerk_mean)=0.131
- `ppo`: low_speed_frac=0.014
  - low_speed_curvature mean=0.000 median=0.000 p90=0.000
  - low_speed_jerk mean=0.000 median=0.000 p90=0.000
  - corr(avg_speed, curvature_mean)=-0.117 corr(avg_speed, jerk_mean)=-0.194
- `socnav_orca`: low_speed_frac=0.014
  - low_speed_curvature mean=0.000 median=0.000 p90=0.000
  - low_speed_jerk mean=0.000 median=0.000 p90=0.000
  - corr(avg_speed, curvature_mean)=-0.635 corr(avg_speed, jerk_mean)=-0.353

Interpretation: fast_pysf_planner shows very large curvature when speed is low (heavy-tailed curvature; negative correlation), suggesting curvature is unstable when near-stationary. PPO shows extremely high mean curvature driven by a few outliers (median is small), so consider filtering curvature/jerk when speed < ε in analysis.

## Frame Snapshots

Sample frames extracted at 25/50/75% of episode duration for top-problem episodes.

### fast_pysf_planner
- `output/analysis/planner_comparison_2026_01_28/frames/fast_pysf_planner/classic_crossing_high_seed231_fast_pysf_planner_t25.png`
- `output/analysis/planner_comparison_2026_01_28/frames/fast_pysf_planner/classic_crossing_high_seed231_fast_pysf_planner_t50.png`
- `output/analysis/planner_comparison_2026_01_28/frames/fast_pysf_planner/classic_crossing_high_seed231_fast_pysf_planner_t75.png`
- `output/analysis/planner_comparison_2026_01_28/frames/fast_pysf_planner/classic_overtaking_medium_seed123_fast_pysf_planner_t25.png`
- `output/analysis/planner_comparison_2026_01_28/frames/fast_pysf_planner/classic_overtaking_medium_seed123_fast_pysf_planner_t50.png`
- `output/analysis/planner_comparison_2026_01_28/frames/fast_pysf_planner/classic_overtaking_medium_seed123_fast_pysf_planner_t75.png`

### ppo
- `output/analysis/planner_comparison_2026_01_28/frames/ppo/classic_overtaking_medium_seed231_ppo_t25.png`
- `output/analysis/planner_comparison_2026_01_28/frames/ppo/classic_overtaking_medium_seed231_ppo_t50.png`
- `output/analysis/planner_comparison_2026_01_28/frames/ppo/classic_overtaking_medium_seed231_ppo_t75.png`
- `output/analysis/planner_comparison_2026_01_28/frames/ppo/francis2023_leave_group_seed777_ppo_t25.png`
- `output/analysis/planner_comparison_2026_01_28/frames/ppo/francis2023_leave_group_seed777_ppo_t50.png`
- `output/analysis/planner_comparison_2026_01_28/frames/ppo/francis2023_leave_group_seed777_ppo_t75.png`

### socnav_orca
- `output/analysis/planner_comparison_2026_01_28/frames/socnav_orca/francis2023_leave_group_seed231_socnav_orca_t25.png`
- `output/analysis/planner_comparison_2026_01_28/frames/socnav_orca/francis2023_leave_group_seed231_socnav_orca_t50.png`
- `output/analysis/planner_comparison_2026_01_28/frames/socnav_orca/francis2023_leave_group_seed231_socnav_orca_t75.png`
- `output/analysis/planner_comparison_2026_01_28/frames/socnav_orca/francis2023_robot_crowding_seed1337_socnav_orca_t25.png`
- `output/analysis/planner_comparison_2026_01_28/frames/socnav_orca/francis2023_robot_crowding_seed1337_socnav_orca_t50.png`
- `output/analysis/planner_comparison_2026_01_28/frames/socnav_orca/francis2023_robot_crowding_seed1337_socnav_orca_t75.png`
