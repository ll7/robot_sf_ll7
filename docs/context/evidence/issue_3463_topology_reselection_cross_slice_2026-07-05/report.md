# Issue 3463 Topology Reselection Cross-Slice Diagnostic

Claim boundary: `diagnostic_only_not_benchmark_or_paper_evidence`.
Classification: `blocked`.

At least one progress-gated row did not produce diagnostic evidence.

## Decision Table

| Slice | Role | Candidate | Threshold | Status | Outcome | Progress m | Switches | Deadlock steps | Collision rate |
|---|---|---|---:|---|---|---:|---:|---:|---:|
| bottleneck_transfer | hard | baseline | NA | diagnostic_complete | success | 6.517157384638592 | 2 | 0 | 0.0 |
| bottleneck_transfer | hard | reuse_penalty | NA | diagnostic_complete | horizon_exhausted | 6.517157384638592 | 3 | 159 | 0.0 |
| bottleneck_transfer | hard | progress_gated | 0.05 | diagnostic_complete | success | 6.517157384638592 | 2 | 0 | 0.0 |
| bottleneck_transfer | hard | progress_gated | 0.1 | diagnostic_complete | success | 6.517157384638592 | 2 | 0 | 0.0 |
| bottleneck_transfer | hard | progress_gated | 0.2 | diagnostic_complete | success | 6.517157384638592 | 2 | 0 | 0.0 |
| doorway_transfer | hard | baseline | NA | not_available | obstacle_collision | -12.02842730398372 | 0 | 0 | 1.0 |
| doorway_transfer | hard | reuse_penalty | NA | not_available | obstacle_collision | -12.02842730398372 | 0 | 0 | 1.0 |
| doorway_transfer | hard | progress_gated | 0.05 | not_available | obstacle_collision | -12.02842730398372 | 0 | 0 | 1.0 |
| doorway_transfer | hard | progress_gated | 0.1 | not_available | obstacle_collision | -12.02842730398372 | 0 | 0 | 1.0 |
| doorway_transfer | hard | progress_gated | 0.2 | not_available | obstacle_collision | -12.02842730398372 | 0 | 0 | 1.0 |
| t_intersection_transfer | hard | baseline | NA | diagnostic_complete | pedestrian_collision | 3.8686292077485884 | 0 | 0 | 1.0 |
| t_intersection_transfer | hard | reuse_penalty | NA | diagnostic_complete | pedestrian_collision | 3.8686292077485884 | 0 | 0 | 1.0 |
| t_intersection_transfer | hard | progress_gated | 0.05 | diagnostic_complete | pedestrian_collision | 3.8686292077485884 | 0 | 0 | 1.0 |
| t_intersection_transfer | hard | progress_gated | 0.1 | diagnostic_complete | pedestrian_collision | 3.8686292077485884 | 0 | 0 | 1.0 |
| t_intersection_transfer | hard | progress_gated | 0.2 | diagnostic_complete | pedestrian_collision | 3.8686292077485884 | 0 | 0 | 1.0 |
| simple_negative_control | negative_control | baseline | NA | diagnostic_complete | success | 1.8000000268220901 | 0 | 0 | 0.0 |
| simple_negative_control | negative_control | reuse_penalty | NA | diagnostic_complete | success | 1.8000000268220901 | 0 | 0 | 0.0 |
| simple_negative_control | negative_control | progress_gated | 0.05 | diagnostic_complete | success | 1.8000000268220901 | 0 | 0 | 0.0 |
| simple_negative_control | negative_control | progress_gated | 0.1 | diagnostic_complete | success | 1.8000000268220901 | 0 | 0 | 0.0 |
| simple_negative_control | negative_control | progress_gated | 0.2 | diagnostic_complete | success | 1.8000000268220901 | 0 | 0 | 0.0 |

## Caveats

- Evidence is diagnostic-only and fail-closed; failed, unavailable, or degraded rows are not success evidence.
- `detour_cost_proxy` is the non-primary topology-command step count, not a path-optimality proof.
- `oscillation_count` uses topology hypothesis switch count from existing diagnostics.
