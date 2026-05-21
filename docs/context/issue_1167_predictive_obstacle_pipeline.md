# Issue #1167 Predictive Obstacle-Feature Pipeline

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1167>

## Decision

The same-seed obstacle-feature comparison should use the existing predictive training pipeline, but
the pipeline must explicitly carry `predictive_obstacle_features_v1` through base collection,
hardcase collection, and training. Before this note, `run_predictive_training_pipeline.py` did not
pass a model family to the collectors or trainer, and the hardcase collector only emitted legacy or
ego-conditioned rows.

This change does not claim an obstacle-feature planner improvement. It makes the config-first
comparison path runnable with real map-derived obstacle rows so the full training/evaluation
campaign can be submitted separately.

## Implemented Contract

- `scripts/training/run_predictive_training_pipeline.py` accepts `model_family` from the top-level
  config, collection config, or training config, then passes it to both collection stages and
  `train_predictive_planner.py`.
- `scripts/training/collect_predictive_hardcase_data.py` now supports
  `--model-family predictive_obstacle_features_v1`, emits `feature_schema_json`, and appends the
  same six map-derived obstacle features used by `collect_predictive_planner_data.py`.
- `configs/training/predictive/predictive_obstacle_features_v1_issue_1167.yaml` records the
  canonical issue #1167 config-first command path.

Canonical command:

```bash
uv run python scripts/training/run_predictive_training_pipeline.py \
  --config configs/training/predictive/predictive_obstacle_features_v1_issue_1167.yaml
```

## Local Smoke Evidence

Hardcase obstacle-row smoke:

```bash
printf 'classic_cross_trap_low: [102]\n' > /tmp/issue1167_seed.yaml
LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 uv run python \
  scripts/training/collect_predictive_hardcase_data.py \
  --scenario-matrix configs/scenarios/classic_interactions.yaml \
  --seed-manifest /tmp/issue1167_seed.yaml \
  --max-steps 8 \
  --horizon-steps 1 \
  --max-agents 8 \
  --model-family predictive_obstacle_features_v1 \
  --output output/tmp/issue1167/hardcase_obstacle_smoke.npz
```

Observed result:

- `num_samples`: `7`
- `state_dim`: `10`
- `model_family`: `predictive_obstacle_features_v1`
- `obstacle_feature_source`: `map_geometry`
- `obstacle_line_count`: `28`
- active valid obstacle rows: `21`
- first valid obstacle row: approximately
  `[4.5137, -0.6086, -0.7935, 1.0, 0.0, 1.0]`

The generated `.npz` and coverage outputs remain under ignored `output/` paths; this note preserves
only compact command/result evidence.

## Validation

Targeted validation:

```bash
uv run ruff check scripts/training/collect_predictive_hardcase_data.py \
  scripts/training/run_predictive_training_pipeline.py \
  tests/training/test_collect_predictive_hardcase_data.py \
  tests/training/test_run_predictive_training_pipeline.py

uv run pytest -q tests/training/test_collect_predictive_hardcase_data.py \
  tests/training/test_run_predictive_training_pipeline.py \
  tests/training/test_collect_predictive_planner_data.py \
  tests/training/test_train_predictive_planner_dataset_diagnostics.py
```

Observed targeted pytest result: `32 passed`.

## Follow-Up Boundary

Issue #1167 is not complete until the same-seed training/evaluation comparison reports ADE/FDE and
downstream planner metrics. This patch removes the local pipeline blocker recorded during PR #1400
triage and should be treated as the runnable setup for the campaign, not as benchmark-success
evidence.
