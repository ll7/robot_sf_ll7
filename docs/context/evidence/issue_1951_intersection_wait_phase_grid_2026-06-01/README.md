# Issue #1951 Intersection-Wait Phase-Grid Evidence

This bundle preserves compact, reviewable evidence for the diagnostic-only
`francis2023_intersection_wait` single-pedestrian phase grid.

## Contents

- `summary.json`: compact tracked summary emitted by
  `scripts/validation/run_scenario_perturbation_criticality_pilot.py`.
- `family_magnitude_summary.json`: compact derived rollup by perturbation family and requested
  magnitude, including the invalid negative start-delay row.
- `SHA256SUMS`: checksums for the tracked compact summaries and this README.

## Boundary

Raw episode JSONL, materialized scenario matrices, and preflight JSON remain ignored under
`output/`. The pilot is local diagnostic evidence only, not benchmark-strength or paper-facing
evidence.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
uv run python scripts/validation/run_scenario_perturbation_criticality_pilot.py \
  configs/scenarios/perturbations/issue_1610_intersection_wait_phase_grid_v1.yaml \
  --materialized-output-dir output/scenario_perturbations/issue1951_intersection_wait_phase_grid/materialized \
  --pilot-output-dir output/scenario_perturbations/issue1951_intersection_wait_phase_grid/pilot \
  --seed-limit 4 \
  --horizon 80 \
  --dt 0.1 \
  --workers 1 \
  --planner goal \
  --planner orca \
  --planner scenario_adaptive_hybrid_orca_v2_collision_guard \
  --evidence-summary docs/context/evidence/issue_1951_intersection_wait_phase_grid_2026-06-01/summary.json
```

## Result

Preflight found 10 manifest variants: one no-op baseline, eight eligible perturbation rows, and
one excluded row. The excluded row was `francis2023_intersection_wait_start_delay_h1_m050`, because
the source pedestrian has no baseline `start_delay_s` and the `-0.5 s` offset would make it
negative.

The pilot ran 81 episodes: 9 materialized variants x 3 seeds x 3 planners. It produced 72/72
completed paired no-op-versus-perturbation rows. Mean outcome deltas were all zero for success,
collision, and timeout. Mean min-distance deltas by requested magnitude:

| Family | Magnitude | Completed Pairs | Mean Min-Distance Delta |
|---|---:|---:|---:|
| start delay | `-0.5 s` | 0 | excluded: negative start delay |
| start delay | `+0.5 s` | 9 | `+4.159358 m` |
| start delay | `+1.0 s` | 9 | `+4.159358 m` |
| speed | `-0.25 m/s` | 9 | `+2.172367 m` |
| speed | `+0.25 m/s` | 9 | `-2.002917 m` |
| speed | `+0.5 m/s` | 9 | `-3.862581 m` |
| wait duration | `-0.5 s` | 9 | `0.0 m` |
| wait duration | `+0.5 s` | 9 | `0.0 m` |
| wait duration | `+1.0 s` | 9 | `0.0 m` |

Interpretation: on this fixed local boundary, speed magnitude shows the clearest signed closest-
clearance response, positive start delays remain beneficial for closest clearance but saturate
between `+0.5 s` and `+1.0 s`, and wait-duration offsets remain flat. This does not establish
planner robustness or global perturbation importance.
