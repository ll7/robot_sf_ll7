# Issue 677 MPPI Social Benchmark Note

## Goal

Benchmark `mppi_social` against the current predictive planner baselines on the canonical paper
surface.

## Why This Was Worth Trying

`mppi_social` keeps the social-cost framing but replaces the search strategy with an MPPI-style
controller. That made it a stronger remaining candidate than the already-failed portfolio and
predictive-MPPI variants because it tests a different optimization family instead of only retuning
an existing predictive stack.

## Configs

- Benchmark config:
  `configs/benchmarks/paper_experiment_matrix_v1_mppi_social_compare.yaml`
- Candidate algo:
  `configs/algos/mppi_social_camera_ready.yaml`
- Baseline algos:
  - `configs/algos/prediction_planner_camera_ready.yaml`
  - `configs/algos/prediction_planner_camera_ready_xl_ego.yaml`

## Validation Command

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_mppi_social_compare.yaml \
  --label issue677_mppi_social_compare \
  --log-level WARNING
```

## Canonical Artifact

- Final campaign:
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_mppi_social_compare_issue677_mppi_social_compare_20260320_201402`

## Final Result

| Planner | Success | Collisions | SNQI | Runtime (s) | Near Misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mppi_social` | `0.0284` | `0.2199` | `-0.3801` | `733.8462` | `2.1348` |
| `prediction_planner_v2_full` | `0.0638` | `0.2270` | `-0.1960` | `165.4703` | `7.5887` |
| `prediction_planner_v2_xl_ego` | `0.0567` | `0.2270` | `-0.1650` | `187.3446` | `4.9645` |

## Interpretation

- `mppi_social` is materially slower than either predictive baseline.
- It succeeds less often than either predictive baseline.
- It does reduce near misses and slightly reduces collision rate versus the predictive baselines.

That safety signal is real, but it is overwhelmed by the runtime cost and by the drop in goal
completion. The SNQI result is also substantially worse than both predictive baselines, which means
it does not improve the overall benchmark tradeoff.

## Verdict

Keep `mppi_social` as a testing-only exploratory planner.

It is not a benchmark-quality upgrade and should not be promoted toward the main planner set.

## Follow-Up Recommendation

Do not spend more time on this exact `mppi_social` configuration unless there is a concrete new
hypothesis about how to preserve its safety signal without paying the current success and runtime
penalties.
