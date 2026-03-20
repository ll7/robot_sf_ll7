# Issue 671 Gap Prediction Benchmark Note

## Goal

Benchmark `gap_prediction` against the current predictive planner baselines on the canonical
paper surface and decide whether it is strong enough to justify more benchmark investment.

## Scope

- compare `gap_prediction` against:
  - `prediction_planner_v2_full`
  - `prediction_planner_v2_xl_ego`
- use the same paper-facing camera-ready campaign contract as the predictive v2 comparison
- end with a benchmark-readiness verdict, not just a smoke-run verdict

## Configs

- Benchmark config:
  `configs/benchmarks/paper_experiment_matrix_v1_gap_prediction_compare.yaml`
- Baseline algos:
  - `configs/algos/prediction_planner_camera_ready.yaml`
  - `configs/algos/prediction_planner_camera_ready_xl_ego.yaml`
- Candidate algo:
  - `configs/algos/gap_prediction_camera_ready.yaml`

## Validation Command

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_gap_prediction_compare.yaml \
  --label issue671_gap_prediction_compare_final \
  --log-level WARNING
```

## Important Execution Note

The first `issue671_gap_prediction_compare` campaign exposed a config-contract bug in this branch:
camera-ready campaign configs must use `algo_config`, not `algo_config_path`.

That bug was fixed before the final result below was recorded.

## Canonical Artifact

- Final campaign:
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_gap_prediction_compare_issue671_gap_prediction_compare_final_20260320_185404`

## Final Result

| Planner | Success | Collisions | SNQI | Runtime (s) | Near Misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gap_prediction` | `0.0000` | `0.0213` | `-0.1215` | `59.3905` | `1.5745` |
| `prediction_planner_v2_full` | `0.0567` | `0.2411` | `-0.1895` | `144.9303` | `7.0000` |
| `prediction_planner_v2_xl_ego` | `0.0709` | `0.2270` | `-0.1638` | `158.1098` | `5.0426` |

## Interpretation

- `gap_prediction` is materially safer than the current predictive baselines.
- It is also materially faster.
- But it never reaches the goal on the paper surface.

That makes the overall result straightforward:

- this planner is behaving like an over-conservative veto layer,
- not like a competitive local planner.

## Why It Is Not Promising As A Headline Planner

The current benchmark target is not just low collision rate. It is a balanced success/safety result
that can compete with the current headline baselines.

`gap_prediction` fails that test:

- zero success means it cannot be treated as a viable local-planner replacement,
- the safety gain is achieved by refusing progress too aggressively,
- the result is weaker than both predictive baselines on goal-reaching behavior.

## Verdict

Keep `gap_prediction` as a testing-only exploratory planner.

Do **not** promote it toward the main benchmark set and do **not** invest further in this exact
configuration as a headline candidate.

## Follow-Up Recommendation

If this family is revisited later, the next work should target the cause of the zero-success mode:

- the interaction between `stream_gap` vetoing and progress recovery,
- not more broad benchmark reruns of the current config.
