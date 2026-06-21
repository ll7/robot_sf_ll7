# Issue #3286 Cross-Benchmark Metric Wrappers

Status: smoke evidence for trace-derived wrappers, not simulator parity or paper-grade evidence.

Related:
- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/3286
- Input correspondence note: `docs/context/issue_2928_socnavbench_hunavsim_metric_correspondence.md`
- Versioned mapping: `configs/benchmarks/cross_benchmark_metric_mapping_v1.yaml`
- Wrapper module: `robot_sf/benchmark/cross_benchmark_metrics.py`

## Claim Boundary

The new wrapper layer exposes Robot SF trace-derived metrics in an external-style row format. It
does not show that Robot SF, SocNavBench, or HuNavSim are equivalent simulators. Rows are labeled as
`available`, `approximate`, or `unavailable`, and unavailable external-only metrics remain in the
report so consumers cannot silently treat missing simulator state as a zero-valued metric.

## Implemented Wrapper Rows

| Metric ID | Source | Status Boundary |
| --- | --- | --- |
| `socnavbench.path_length_ratio` | `socnavbench_path_length_ratio` | available when the Robot SF trajectory and goal displacement are valid |
| `common.traversal_time_s` | `time_to_goal` | approximate because timeout and episode-stop semantics vary by benchmark |
| `common.time_to_collision_min_s` | `time_to_collision_min` | approximate because it uses a constant-velocity approaching-pair assumption |
| `common.closest_pedestrian_distance_m` | `distance_to_human_min` | available as center-to-center distance, not footprint clearance |
| `robot_sf.success_trace_predicate` | `success` | available when a horizon is supplied |
| `socnavbench.personal_space_objective` | none | unavailable; external objective field is not present in Robot SF traces |
| `hunavsim.human_behavior_cost` | none | unavailable; HuNavSim behavior-model state is not present in Robot SF traces |

## Fixture Smoke Result

The focused fixture test builds a three-step Robot SF trace with one static pedestrian and verifies
that the wrapper report includes:

- available rows for path-length ratio, closest-pedestrian distance, and Robot SF success;
- approximate rows for traversal time and minimum time-to-collision;
- unavailable rows for external simulator-only metrics.

This is smoke evidence that the wrapper contract works on a deterministic trace. It is not a
benchmark result and should not be used as cross-simulator parity evidence.
