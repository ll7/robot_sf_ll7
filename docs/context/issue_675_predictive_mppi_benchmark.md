# Issue 675 Predictive MPPI Benchmark Note

## Goal

Benchmark `predictive_mppi` against the current predictive planner baselines on the canonical
paper surface.

## Why This Was A Reasonable Candidate

`predictive_mppi` changes the control-search method itself while staying inside the predictive-model
family. That made it more promising than the recently tested veto- and portfolio-style candidates.

## Configs

- Benchmark config:
  `configs/benchmarks/paper_experiment_matrix_v1_predictive_mppi_compare.yaml`
- Candidate algo:
  `configs/algos/predictive_mppi_camera_ready.yaml`
- Baseline algos:
  - `configs/algos/prediction_planner_camera_ready.yaml`
  - `configs/algos/prediction_planner_camera_ready_xl_ego.yaml`

## Validation Command

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_predictive_mppi_compare.yaml \
  --label issue675_predictive_mppi_compare \
  --log-level WARNING
```

## Canonical Artifact

- Final campaign:
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_predictive_mppi_compare_issue675_predictive_mppi_compare_20260320_192902`

## Final Result

| Planner | Success | Collisions | SNQI | Runtime (s) | Near Misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prediction_planner_v2_full` | `0.0638` | `0.2128` | `-0.1902` | `159.3323` | `7.7163` |
| `prediction_planner_v2_xl_ego` | `0.0709` | `0.2199` | `-0.1693` | `176.2394` | `5.7801` |
| `predictive_mppi` | `0.0355` | `0.2553` | `-0.1993` | `1223.0338` | `4.9929` |

## Interpretation

- `predictive_mppi` is slower by roughly an order of magnitude.
- It succeeds less often than either predictive baseline.
- It also collides more often than either predictive baseline.

The only mild positive signal is a lower near-miss rate than the plain predictive baseline, but
that is nowhere near enough to offset the success, collision, and runtime regressions.

## Verdict

Keep `predictive_mppi` as a testing-only exploratory planner.

It is not a benchmark-quality upgrade and should not be promoted toward the main planner set.

## Follow-Up Recommendation

Do not spend more time on the current `predictive_mppi` configuration unless there is a concrete
new hypothesis about why its control search is wasting so much runtime without improving outcomes.
