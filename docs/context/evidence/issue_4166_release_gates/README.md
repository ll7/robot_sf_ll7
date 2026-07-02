# Paired Safety And Comfort Release-Gate Matrix

Release gates are evaluated over existing benchmark rows only. Provisional thresholds are not certification or paper-grade approval.

## Pass/Fail Matrix

| Planner | Scenario family | Safety | Comfort | Overall | Failed gates | Not evaluable |
| --- | --- | --- | --- | --- | --- | --- |
| goal | classic_crossing | pass | pass | pass |  |  |
| orca | classic_crossing | fail | pass | fail | collision_rate_zero |  |
| ppo | classic_crossing | pass | fail | fail | proxemic_intrusion_rate_limit;jerk_mean_limit |  |
| prediction_planner | classic_crossing | fail | pass | fail | near_miss_rate_limit;min_clearance_floor |  |
| sacadrl | classic_crossing | pass | pass | pass |  |  |
| social_force | classic_crossing | pass | pass | pass |  |  |
| socnav_bench | classic_crossing | not_evaluable | not_evaluable | not_evaluable |  | collision_rate_zero;near_miss_rate_limit;min_clearance_floor;proxemic_intrusion_rate_limit;jerk_mean_limit |
| socnav_sampling | classic_crossing | pass | not_evaluable | not_evaluable |  | proxemic_intrusion_rate_limit;jerk_mean_limit |

## Failed Or Not-Evaluable Gates

### orca / classic_crossing
- `collision_rate_zero`: fail; metric `collision_rate` observed=[0.1], threshold=max 0.0; provenance=provisional_not_certification_issue_4166
### ppo / classic_crossing
- `proxemic_intrusion_rate_limit`: fail; metric `proxemic_intrusion_rate` observed=[0.13], threshold=max 0.1; provenance=provisional_not_certification_issue_4166
- `jerk_mean_limit`: fail; metric `jerk_mean` observed=[2.4], threshold=max 2.0; provenance=provisional_not_certification_issue_4166
### prediction_planner / classic_crossing
- `near_miss_rate_limit`: fail; metric `near_miss_rate` observed=[0.07], threshold=max 0.05; provenance=provisional_not_certification_issue_4166
- `min_clearance_floor`: fail; metric `min_clearance_m` observed=[0.2], threshold=min 0.25; provenance=provisional_not_certification_issue_4166
### socnav_bench / classic_crossing
- `collision_rate_zero`: not_evaluable; metric `collision_rate` observed=[], threshold=max 0.0; provenance=provisional_not_certification_issue_4166
- `near_miss_rate_limit`: not_evaluable; metric `near_miss_rate` observed=[], threshold=max 0.05; provenance=provisional_not_certification_issue_4166
- `min_clearance_floor`: not_evaluable; metric `min_clearance_m` observed=[], threshold=min 0.25; provenance=provisional_not_certification_issue_4166
- `proxemic_intrusion_rate_limit`: not_evaluable; metric `proxemic_intrusion_rate` observed=[], threshold=max 0.1; provenance=provisional_not_certification_issue_4166
- `jerk_mean_limit`: not_evaluable; metric `jerk_mean` observed=[], threshold=max 2.0; provenance=provisional_not_certification_issue_4166
### socnav_sampling / classic_crossing
- `proxemic_intrusion_rate_limit`: not_evaluable; metric `proxemic_intrusion_rate` observed=[], threshold=max 0.1; provenance=provisional_not_certification_issue_4166
- `jerk_mean_limit`: not_evaluable; metric `jerk_mean` observed=[], threshold=max 2.0; provenance=provisional_not_certification_issue_4166

## Threshold Provenance

- `collision_rate_zero` (safety, `collision_rate`): max 0.0; provisional_not_certification_issue_4166
- `near_miss_rate_limit` (safety, `near_miss_rate`): max 0.05; provisional_not_certification_issue_4166
- `min_clearance_floor` (safety, `min_clearance_m`): min 0.25; provisional_not_certification_issue_4166
- `proxemic_intrusion_rate_limit` (comfort, `proxemic_intrusion_rate`): max 0.1; provisional_not_certification_issue_4166
- `jerk_mean_limit` (comfort, `jerk_mean`): max 2.0; provisional_not_certification_issue_4166
