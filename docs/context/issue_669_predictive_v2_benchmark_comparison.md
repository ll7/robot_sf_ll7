## Issue #669 Prediction Planner v2 Benchmark Comparison

### Goal

Compare the ego-conditioned predictive planner v2 checkpoint path against the current
camera-ready `prediction_planner` baseline on the canonical paper surface.

### Inputs

- Baseline algo config: `configs/algos/prediction_planner_camera_ready.yaml`
- Ego-conditioned algo config: `configs/algos/prediction_planner_camera_ready_xl_ego.yaml`
- Benchmark config:
  `configs/benchmarks/paper_experiment_matrix_v1_prediction_planner_v2_compare.yaml`
- Canonical comparison surface:
  `configs/scenarios/classic_interactions_francis2023.yaml`

### Validation Commands

```bash
uv run python scripts/validation/run_predictive_hard_seed_diagnostics.py \
  --scenario-matrix configs/scenarios/classic_interactions.yaml \
  --seed-manifest configs/benchmarks/predictive_hard_seeds_v1.yaml \
  --checkpoint output/tmp/predictive_planner/pipeline/predictive_br07_all_maps_randomized_full_20260305T123116Z/training/predictive_model.pt \
  --output-dir output/tmp/predictive_planner/diagnostics/issue669_baseline

uv run python scripts/validation/run_predictive_hard_seed_diagnostics.py \
  --scenario-matrix configs/scenarios/classic_interactions.yaml \
  --seed-manifest configs/benchmarks/predictive_hard_seeds_v1.yaml \
  --checkpoint output/model_cache/predictive_proxy_selected_v2_xl_ego/predictive_model.pt \
  --algo-params configs/algos/prediction_planner_camera_ready_xl_ego.yaml \
  --output-dir output/tmp/predictive_planner/diagnostics/issue669_xl_ego

uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_prediction_planner_v2_compare.yaml \
  --label issue669_prediction_planner_v2_compare \
  --log-level WARNING
```

### Artifacts

- Hard-seed baseline:
  `output/tmp/predictive_planner/diagnostics/issue669_baseline/hard_seed_diagnostics_summary.json`
- Hard-seed ego-conditioned:
  `output/tmp/predictive_planner/diagnostics/issue669_xl_ego/hard_seed_diagnostics_summary.json`
- Paper-surface campaign:
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_prediction_planner_v2_compare_issue669_prediction_planner_v2_compare_20260320_164916`

### Hard-Seed Result

- Baseline:
  - `1` success / `6` failures
  - `5` episodes with pedestrian-collision flags
- Ego-conditioned:
  - `1` success / `6` failures
  - `3` episodes with pedestrian-collision flags

Interpretation:

- The ego-conditioned model reduced pedestrian-collision-flag episodes on the hard seeds.
- It did not convert that safety gain into more successes on the hard-seed set.

### Paper-Surface Result

| Planner | Success | Collisions | SNQI | Runtime (s) | Near Misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prediction_planner_v2_full` | `0.0567` | `0.2199` | `-0.1909` | `147.6889` | `7.8085` |
| `prediction_planner_v2_xl_ego` | `0.0638` | `0.2270` | `-0.1681` | `160.3473` | `5.4610` |

Delta (`xl_ego - full`):

- success: `+0.0071`
- collisions: `+0.0071`
- SNQI: `+0.0228`
- runtime: `+12.6584s`
- near misses: `-2.3475`
- comfort exposure: `-0.0029`
- jerk: `+0.0056`

Interpretation:

- The ego-conditioned v2 planner is slightly better on success and SNQI.
- It also collides slightly more and runs slower.
- The net effect is a mild tradeoff, not a clear planner-quality win.

### Position Relative To Current Headline Baselines

Canonical issue-579 paper surface for context:

| Planner | Success | Collisions | SNQI |
| --- | ---: | ---: | ---: |
| `ppo` | `0.2695` | `0.1844` | `-0.3541` |
| `orca` | `0.2340` | `0.0426` | `-0.2325` |
| `prediction_planner` | `0.0709` | `0.2128` | `-0.1924` |

Interpretation:

- Even with the ego-conditioned checkpoint, the predictive planner family remains far below
  the current `ppo` and `orca` success rates.
- This is not the planner-quality breakthrough needed to challenge the current best planners.

### Verdict

`prediction_planner_v2_xl_ego` is a valid comparison candidate and modestly improves some
benchmark-facing signals over the current predictive baseline, but it is not a compelling
replacement and does not outperform the repository's current strongest planners.

### Follow-Up Risk

- Further predictive-model work should target planner-level decision quality, not just
  predictor fidelity.
- If work continues on this family, the next useful issue should focus on why reduced
  near-miss and hard-seed collision signals are not translating into materially better
  goal-reaching behavior.
