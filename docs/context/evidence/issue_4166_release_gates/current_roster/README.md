# Paired Safety And Comfort Release-Gate Matrix

Release gates are evaluated over existing benchmark rows only. Provisional thresholds are not certification or paper-grade approval.

## Pass/Fail Matrix

| Planner | Scenario family | Safety | Comfort | Overall | Failed gates | Not evaluable |
| --- | --- | --- | --- | --- | --- | --- |
| goal | all | fail | pass | fail | mean_collisions_budget | min_clearance_floor_coverage_gap;proxemic_intrusion_rate_coverage_gap |
| orca | all | pass | pass | pass |  | min_clearance_floor_coverage_gap;proxemic_intrusion_rate_coverage_gap |
| ppo | all | pass | pass | pass |  | min_clearance_floor_coverage_gap;proxemic_intrusion_rate_coverage_gap |
| prediction_planner | all | fail | pass | fail | mean_collisions_budget;mean_near_misses_budget | min_clearance_floor_coverage_gap;proxemic_intrusion_rate_coverage_gap |
| sacadrl | all | fail | pass | fail | mean_collisions_budget | min_clearance_floor_coverage_gap;proxemic_intrusion_rate_coverage_gap |
| social_force | all | fail | pass | fail | mean_collisions_budget | min_clearance_floor_coverage_gap;proxemic_intrusion_rate_coverage_gap |
| socnav_bench | all | not_evaluable | not_evaluable | not_evaluable |  | mean_collisions_budget;mean_near_misses_budget;min_clearance_floor_coverage_gap;jerk_mean_limit;comfort_exposure_limit;proxemic_intrusion_rate_coverage_gap |
| socnav_sampling | all | fail | pass | fail | mean_collisions_budget | min_clearance_floor_coverage_gap;proxemic_intrusion_rate_coverage_gap |

## Failed Or Not-Evaluable Gates

### goal / all
- `mean_collisions_budget`: fail; metric `collisions_mean` observed=[0.2361], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_mean_collisions_per_episode
- `min_clearance_floor_coverage_gap`: not_evaluable; metric `min_clearance_m` observed=[], threshold=min 0.25; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
- `proxemic_intrusion_rate_coverage_gap`: not_evaluable; metric `proxemic_intrusion_rate` observed=[], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
### orca / all
- `min_clearance_floor_coverage_gap`: not_evaluable; metric `min_clearance_m` observed=[], threshold=min 0.25; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
- `proxemic_intrusion_rate_coverage_gap`: not_evaluable; metric `proxemic_intrusion_rate` observed=[], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
### ppo / all
- `min_clearance_floor_coverage_gap`: not_evaluable; metric `min_clearance_m` observed=[], threshold=min 0.25; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
- `proxemic_intrusion_rate_coverage_gap`: not_evaluable; metric `proxemic_intrusion_rate` observed=[], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
### prediction_planner / all
- `mean_collisions_budget`: fail; metric `collisions_mean` observed=[0.2083], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_mean_collisions_per_episode
- `mean_near_misses_budget`: fail; metric `near_misses_mean` observed=[8.3681], threshold=max 5.0; provenance=provisional_not_certification_issue_4166_mean_near_misses_per_episode
- `min_clearance_floor_coverage_gap`: not_evaluable; metric `min_clearance_m` observed=[], threshold=min 0.25; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
- `proxemic_intrusion_rate_coverage_gap`: not_evaluable; metric `proxemic_intrusion_rate` observed=[], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
### sacadrl / all
- `mean_collisions_budget`: fail; metric `collisions_mean` observed=[0.3889], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_mean_collisions_per_episode
- `min_clearance_floor_coverage_gap`: not_evaluable; metric `min_clearance_m` observed=[], threshold=min 0.25; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
- `proxemic_intrusion_rate_coverage_gap`: not_evaluable; metric `proxemic_intrusion_rate` observed=[], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
### social_force / all
- `mean_collisions_budget`: fail; metric `collisions_mean` observed=[0.2083], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_mean_collisions_per_episode
- `min_clearance_floor_coverage_gap`: not_evaluable; metric `min_clearance_m` observed=[], threshold=min 0.25; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
- `proxemic_intrusion_rate_coverage_gap`: not_evaluable; metric `proxemic_intrusion_rate` observed=[], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
### socnav_bench / all
- `mean_collisions_budget`: not_evaluable; metric `collisions_mean` observed=[], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_mean_collisions_per_episode
- `mean_near_misses_budget`: not_evaluable; metric `near_misses_mean` observed=[], threshold=max 5.0; provenance=provisional_not_certification_issue_4166_mean_near_misses_per_episode
- `min_clearance_floor_coverage_gap`: not_evaluable; metric `min_clearance_m` observed=[], threshold=min 0.25; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
- `jerk_mean_limit`: not_evaluable; metric `jerk_mean` observed=[], threshold=max 2.0; provenance=provisional_not_certification_issue_4166
- `comfort_exposure_limit`: not_evaluable; metric `comfort_exposure_mean` observed=[], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_proxemic_exposure_proxy
- `proxemic_intrusion_rate_coverage_gap`: not_evaluable; metric `proxemic_intrusion_rate` observed=[], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
### socnav_sampling / all
- `mean_collisions_budget`: fail; metric `collisions_mean` observed=[0.5278], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_mean_collisions_per_episode
- `min_clearance_floor_coverage_gap`: not_evaluable; metric `min_clearance_m` observed=[], threshold=min 0.25; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
- `proxemic_intrusion_rate_coverage_gap`: not_evaluable; metric `proxemic_intrusion_rate` observed=[], threshold=max 0.1; provenance=provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready

## Threshold Provenance

- `mean_collisions_budget` (safety, `collisions_mean`): max 0.1; provisional_not_certification_issue_4166_mean_collisions_per_episode
- `mean_near_misses_budget` (safety, `near_misses_mean`): max 5.0; provisional_not_certification_issue_4166_mean_near_misses_per_episode
- `min_clearance_floor_coverage_gap` (safety, `min_clearance_m`): min 0.25; provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
- `jerk_mean_limit` (comfort, `jerk_mean`): max 2.0; provisional_not_certification_issue_4166
- `comfort_exposure_limit` (comfort, `comfort_exposure_mean`): max 0.1; provisional_not_certification_issue_4166_proxemic_exposure_proxy
- `proxemic_intrusion_rate_coverage_gap` (comfort, `proxemic_intrusion_rate`): max 0.1; provisional_not_certification_issue_4166_metric_not_recorded_by_camera_ready
