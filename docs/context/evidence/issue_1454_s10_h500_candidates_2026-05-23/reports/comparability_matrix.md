# Alyassi Comparability Summary

- Mapping path: `configs/benchmarks/alyassi_comparability_map_v1.yaml`
- Mapping version: `alyassi-comparability-v1`
- Mapping hash: `6f349046993d`

## Coverage Overlap Matrix

| Robot SF Family | Scenario Count | Alyassi Category | Overlap |
|---|---:|---|---|
| accompanying_peer | 1 | unmapped | amv_extension |
| blind_corner | 1 | unmapped | amv_extension |
| bottleneck | 4 | unmapped | amv_extension |
| circular_crossing | 1 | unmapped | amv_extension |
| cross_trap | 3 | unmapped | amv_extension |
| crossing | 1 | unmapped | amv_extension |
| crowd_navigation | 1 | unmapped | amv_extension |
| doorway | 3 | unmapped | amv_extension |
| down_path | 1 | unmapped | amv_extension |
| entering_elevator | 1 | unmapped | amv_extension |
| entering_room | 1 | unmapped | amv_extension |
| exiting_elevator | 1 | unmapped | amv_extension |
| exiting_room | 1 | unmapped | amv_extension |
| following_human | 1 | unmapped | amv_extension |
| frontal_approach | 1 | unmapped | amv_extension |
| group_crossing | 3 | unmapped | amv_extension |
| head_on_corridor | 2 | unmapped | amv_extension |
| intersection_no_gesture | 1 | unmapped | amv_extension |
| intersection_proceed | 1 | unmapped | amv_extension |
| intersection_wait | 1 | unmapped | amv_extension |
| join_group | 1 | unmapped | amv_extension |
| leading_human | 1 | unmapped | amv_extension |
| leave_group | 1 | unmapped | amv_extension |
| merging | 2 | unmapped | amv_extension |
| narrow_doorway | 1 | unmapped | amv_extension |
| narrow_hallway | 1 | unmapped | amv_extension |
| overtaking | 2 | unmapped | amv_extension |
| parallel_traffic | 1 | unmapped | amv_extension |
| pedestrian_obstruction | 1 | unmapped | amv_extension |
| pedestrian_overtaking | 1 | unmapped | amv_extension |
| perpendicular_traffic | 1 | unmapped | amv_extension |
| robot_crowding | 1 | unmapped | amv_extension |
| robot_overtaking | 1 | unmapped | amv_extension |
| station_platform | 1 | unmapped | amv_extension |
| t_intersection | 2 | unmapped | amv_extension |

## Metric Comparability

| Metric | Classification | Alyassi Metric | Rationale |
|---|---|---|---|
| collisions | comparable | collision_rate | Collision rate is directly comparable at episode granularity. |
| comfort_exposure | proxy | comfort/smoothness terms | Comfort exposure overlaps with smoothness-focused safety comfort metrics. |
| near_misses | proxy | safety-distance violations | Near misses approximate low-clearance safety stress but are not identical. |
| snqi | amv_specific | n/a | SNQI is a robot_sf composite quality index used as an AMV extension metric. |
| success | comparable | success_rate | Both benchmarks report episode-level success completion. |

## AMV-Specific Extensions

- shared-space micromobility interactions
- delivery robot curb-side handoff approach
- mixed sidewalk and bike-lane transitions
