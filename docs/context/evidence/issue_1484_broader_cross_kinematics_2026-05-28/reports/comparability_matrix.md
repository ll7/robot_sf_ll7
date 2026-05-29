# Alyassi Comparability Summary

- Mapping path: `configs/benchmarks/alyassi_comparability_map_v1.yaml`
- Mapping version: `alyassi-comparability-v1`
- Mapping hash: `6f349046993d`

## Coverage Overlap Matrix

| Robot SF Family | Scenario Count | Alyassi Category | Overlap |
|---|---:|---|---|
| cross_trap | 1 | unmapped | amv_extension |

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
