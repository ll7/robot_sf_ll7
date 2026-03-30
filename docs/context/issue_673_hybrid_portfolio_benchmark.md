# Issue 673 Hybrid Portfolio Benchmark Note

## Goal

Benchmark `hybrid_portfolio` against the current predictive planner baselines on the canonical
paper surface.

## Why This Was Worth Trying

Unlike `gap_prediction`, `hybrid_portfolio` can switch between multiple sub-planners instead of
acting only as a conservative veto layer. That made it the strongest remaining testing-only
candidate in the current in-repo portfolio.

## Configs

- Benchmark config:
  `configs/benchmarks/paper_experiment_matrix_v1_hybrid_portfolio_compare.yaml`
- Candidate algo:
  `configs/algos/hybrid_portfolio_camera_ready.yaml`
- Baseline algos:
  - `configs/algos/prediction_planner_camera_ready.yaml`
  - `configs/algos/prediction_planner_camera_ready_xl_ego.yaml`

## Validation Command

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_hybrid_portfolio_compare.yaml \
  --label issue673_hybrid_portfolio_compare \
  --log-level WARNING
```

## Canonical Artifact

- Final campaign:
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_hybrid_portfolio_compare_issue673_hybrid_portfolio_compare_20260320_190520`

## Final Result

| Planner | Success | Collisions | SNQI | Runtime (s) | Near Misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hybrid_portfolio` | `0.0213` | `0.2482` | `-0.3963` | `691.3955` | `3.4326` |
| `prediction_planner_v2_full` | `0.0567` | `0.2270` | `-0.1892` | `145.4059` | `7.0851` |
| `prediction_planner_v2_xl_ego` | `0.0567` | `0.2340` | `-0.1700` | `161.7385` | `5.5674` |

## Interpretation

- `hybrid_portfolio` is slower by a large margin.
- It succeeds less often than either predictive baseline.
- It also collides more than either predictive baseline.

The near-miss reduction is real but not useful enough to offset the drop in success and the large
runtime increase.

## Verdict

Keep `hybrid_portfolio` as a testing-only exploratory planner.

It is neither a credible benchmark upgrade nor a serious candidate to challenge the current
headline planners.

## Follow-Up Recommendation

Do not spend more time on this exact `hybrid_portfolio` configuration.

If this family is revisited, the next work should focus on understanding why the planner-switching
logic loses both success and runtime efficiency instead of assuming that more benchmark reruns will
change the conclusion.
