# Issue 681 Stream Gap Benchmark Note

## Goal

Benchmark `stream_gap` against the current predictive planner baselines on the canonical paper
surface.

## Why This Was Worth Trying

`stream_gap` targets corridor and crossing-style gap acceptance directly instead of relying on the
more expensive predictive-rollout or MPPI families. That made it the cleanest remaining structured
reactive candidate in the current in-repo planner set.

## Configs

- Benchmark config:
  `configs/benchmarks/paper_experiment_matrix_v1_stream_gap_compare.yaml`
- Candidate algo:
  `configs/algos/stream_gap_camera_ready.yaml`
- Baseline algos:
  - `configs/algos/prediction_planner_camera_ready.yaml`
  - `configs/algos/prediction_planner_camera_ready_xl_ego.yaml`

## Validation Command

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_stream_gap_compare.yaml \
  --label issue681_stream_gap_compare \
  --log-level WARNING
```

## Canonical Artifact

- Final campaign:
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_stream_gap_compare_issue681_stream_gap_compare_20260321_150507`

## Final Result

| Planner | Success | Collisions | SNQI | Runtime (s) | Near Misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prediction_planner_v2_full` | `0.0709` | `0.2340` | `-0.1935` | `184.5363` | `7.4255` |
| `prediction_planner_v2_xl_ego` | `0.0709` | `0.2128` | `-0.1665` | `185.3000` | `5.2482` |
| `stream_gap` | `0.0000` | `0.0213` | `-0.1212` | `31.0665` | `1.5745` |

## Interpretation

- `stream_gap` is much faster than both predictive baselines.
- It is also much safer on collisions and near misses.
- But it completely fails on goal-reaching.

This is the same basic failure mode already seen in `gap_prediction`: the planner is conservative
and orderly, but too reluctant to commit to motion through the full paper-surface scenario set.
That keeps the aggregate safety metrics low while collapsing success to zero.

## Verdict

Keep `stream_gap` as a testing-only exploratory planner.

It is not a benchmark-quality upgrade and should not be promoted toward the main planner set.

## Follow-Up Recommendation

Do not spend more time on this exact `stream_gap` configuration unless there is a concrete
hypothesis for recovering commitment/progress behavior without losing the strong safety signal.
