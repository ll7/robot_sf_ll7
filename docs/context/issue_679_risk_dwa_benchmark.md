# Issue 679 Risk DWA Benchmark Note

## Goal

Benchmark `risk_dwa` against the current predictive planner baselines on the canonical paper
surface.

## Why This Was Worth Trying

`risk_dwa` is a simpler reactive local planner with explicit progress, heading, clearance, and TTC
terms. That made it a useful check on whether a lightweight non-learning planner could recover a
better success/safety/runtime tradeoff than the recent testing-only predictive variants.

## Configs

- Benchmark config:
  `configs/benchmarks/paper_experiment_matrix_v1_risk_dwa_compare.yaml`
- Candidate algo:
  `configs/algos/risk_dwa_camera_ready.yaml`
- Baseline algos:
  - `configs/algos/prediction_planner_camera_ready.yaml`
  - `configs/algos/prediction_planner_camera_ready_xl_ego.yaml`

## Validation Command

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_risk_dwa_compare.yaml \
  --label issue679_risk_dwa_compare \
  --log-level WARNING
```

## Canonical Artifact

- Final campaign:
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_risk_dwa_compare_issue679_risk_dwa_compare_20260321_144439`

## Final Result

| Planner | Success | Collisions | SNQI | Runtime (s) | Near Misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prediction_planner_v2_full` | `0.0780` | `0.1986` | `-0.1838` | `179.8297` | `7.6028` |
| `prediction_planner_v2_xl_ego` | `0.0567` | `0.2340` | `-0.1673` | `173.8857` | `5.1773` |
| `risk_dwa` | `0.0142` | `0.2908` | `-0.1711` | `102.0775` | `4.3972` |

## Interpretation

- `risk_dwa` is faster than both predictive baselines.
- It also reduces near misses relative to both predictive baselines.
- That benefit is offset by a 75%+ drop in success versus both predictive baselines and a clear collision increase.

The SNQI lands close to `prediction_planner_v2_xl_ego` only because the faster runtime and lower
near-miss profile partially compensate for the poor outcome counts. That is not a trustworthy sign
of planner quality because the success and collision regressions are too large.

## Verdict

Keep `risk_dwa` as a testing-only exploratory planner.

It is not a benchmark-quality upgrade and should not be promoted toward the main planner set.

## Follow-Up Recommendation

Do not spend more time on this exact `risk_dwa` configuration unless there is a concrete hypothesis
for recovering goal-reaching without losing the reactive safety behavior that lowered near misses.
