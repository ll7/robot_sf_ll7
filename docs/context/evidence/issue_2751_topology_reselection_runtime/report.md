# Issue #2751 Topology Reselection Cross-Slice Diagnostic (2026-06-13)

Claim boundary: `diagnostic_only_not_benchmark_or_paper_evidence`.
Classification: `revise`.

Hard progress-gated rows retained diagnostic influence but all remained horizon_exhausted.

## Decision Table

| Slice | Role | Candidate | Threshold | Status | Outcome | Progress m | Switches | Deadlock steps | Collision rate |
|---|---|---|---:|---|---|---:|---:|---:|---:|
| bottleneck_transfer | hard | baseline | NA | diagnostic_complete | horizon_exhausted | 5.200000077486036 | 0 | 159 | 0.0 |
| bottleneck_transfer | hard | reuse_penalty | NA | diagnostic_complete | horizon_exhausted | 6.117157378678126 | 2 | 159 | 0.0 |
| bottleneck_transfer | hard | progress_gated | 0.05 | diagnostic_complete | horizon_exhausted | 5.200000077486036 | 0 | 159 | 0.0 |
| bottleneck_transfer | hard | progress_gated | 0.1 | diagnostic_complete | horizon_exhausted | 5.400000080466269 | 0 | 159 | 0.0 |
| bottleneck_transfer | hard | progress_gated | 0.2 | diagnostic_complete | horizon_exhausted | 5.600000083446501 | 0 | 159 | 0.0 |
| doorway_transfer | hard | baseline | NA | diagnostic_complete | horizon_exhausted | 1.4485281590086068 | 0 | 159 | 0.0 |
| doorway_transfer | hard | reuse_penalty | NA | diagnostic_complete | horizon_exhausted | 1.4485281590086068 | 0 | 159 | 0.0 |
| doorway_transfer | hard | progress_gated | 0.05 | diagnostic_complete | horizon_exhausted | 1.4485281590086068 | 0 | 159 | 0.0 |
| doorway_transfer | hard | progress_gated | 0.1 | diagnostic_complete | horizon_exhausted | 1.4485281590086068 | 0 | 159 | 0.0 |
| doorway_transfer | hard | progress_gated | 0.2 | diagnostic_complete | horizon_exhausted | 1.4485281590086068 | 0 | 159 | 0.0 |
| t_intersection_transfer | hard | baseline | NA | diagnostic_complete | horizon_exhausted | 4.868629222649751 | 3 | 159 | 0.0 |
| t_intersection_transfer | hard | reuse_penalty | NA | diagnostic_complete | horizon_exhausted | 6.3857865327825385 | 13 | 159 | 0.0 |
| t_intersection_transfer | hard | progress_gated | 0.05 | diagnostic_complete | horizon_exhausted | 6.3857865327825385 | 13 | 159 | 0.0 |
| t_intersection_transfer | hard | progress_gated | 0.1 | diagnostic_complete | horizon_exhausted | 6.3857865327825385 | 13 | 159 | 0.0 |
| t_intersection_transfer | hard | progress_gated | 0.2 | diagnostic_complete | horizon_exhausted | 6.3857865327825385 | 13 | 159 | 0.0 |
| simple_negative_control | negative_control | baseline | NA | diagnostic_complete | success | 1.7656854512600013 | 0 | 0 | 0.0 |
| simple_negative_control | negative_control | reuse_penalty | NA | diagnostic_complete | success | 1.7656854512600013 | 0 | 0 | 0.0 |
| simple_negative_control | negative_control | progress_gated | 0.05 | diagnostic_complete | success | 1.7656854512600013 | 0 | 0 | 0.0 |
| simple_negative_control | negative_control | progress_gated | 0.1 | diagnostic_complete | success | 1.7656854512600013 | 0 | 0 | 0.0 |
| simple_negative_control | negative_control | progress_gated | 0.2 | diagnostic_complete | success | 1.7656854512600013 | 0 | 0 | 0.0 |

## Caveats

- Evidence is diagnostic-only and fail-closed; failed, unavailable, or degraded rows are not success evidence.
- `detour_cost_proxy` is the non-primary topology-command step count, not a path-optimality proof.
- `oscillation_count` uses topology hypothesis switch count from existing diagnostics.
