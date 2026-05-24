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
- `scripts/training/run_predictive_training_pipeline.py` also accepts an optional committed
  `base_collection.seed_manifest` and records the resolved base-seed manifest in the pipeline
  summary so paired same-seed runs do not depend on regenerated output-local manifests.
- `scripts/training/collect_predictive_hardcase_data.py` now supports
  `--model-family predictive_obstacle_features_v1`, emits `feature_schema_json`, and appends the
  same six map-derived obstacle features used by `collect_predictive_planner_data.py`.
- `configs/training/predictive/predictive_obstacle_features_v1_issue_1167.yaml` records the
  canonical issue #1167 config-first command path.
- Obstacle-feature pipeline runs now write `obstacle_feature_preflight.json` and fail before
  training if either collected dataset contains only sentinel obstacle rows instead of any active
  map-derived obstacle features.

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

## Issue #1427 Same-Seed Handoff

Issue #1427 needs runnable handoff surfaces for the actual bounded same-seed comparison, not a new
pipeline design. The committed handoff set is:

- shared base seed manifest:
  `configs/training/predictive/predictive_same_seed_issue_1427_base_seed_manifest.yaml`
- baseline config:
  `configs/training/predictive/predictive_br07_same_seed_issue_1427.yaml`
- obstacle-feature config:
  `configs/training/predictive/predictive_obstacle_features_same_seed_issue_1427.yaml`
- compact evidence manifest:
  `docs/context/evidence/issue_1427_predictive_same_seed_handoff_2026-05-21/manifest.json`

Canonical command pair for external execution:

```bash
uv run python scripts/training/run_predictive_training_pipeline.py \
  --config configs/training/predictive/predictive_br07_same_seed_issue_1427.yaml

uv run python scripts/training/run_predictive_training_pipeline.py \
  --config configs/training/predictive/predictive_obstacle_features_same_seed_issue_1427.yaml
```

Same-seed contract:

- Both variants use the same committed base seed manifest and the same hard-seed manifest.
- Training budget, scenario matrix, planner grid, and evaluation surface are identical between the
  two configs.
- The obstacle-feature run fails closed before training if the generated base or hardcase dataset
  contains zero active non-sentinel obstacle rows. The preflight evidence is written to
  `output/tmp/predictive_planner/pipeline/<run_id>/obstacle_feature_preflight.json`.

## Issue #1427 SLURM Launch Update, 2026-05-24

PR #1480 added `SLURM/Auxme/predictive_training_pipeline.sl` as the Auxme launcher for the
same-seed pair and hardened `SLURM/Auxme/camera_ready_benchmark.sl` against the broken `srun`
symbol-error path observed in earlier campaign submissions.

Submitted jobs from branch `issue-1427-predictive-slurm-runs`:

- job `12611`, `pred1427-base`, config
  `configs/training/predictive/predictive_br07_same_seed_issue_1427.yaml`, run id
  `predictive_br07_same_seed_issue_1427_20260524T110529Z`
- job `12612`, `pred1427-obs`, config
  `configs/training/predictive/predictive_obstacle_features_same_seed_issue_1427.yaml`, run id
  `predictive_obstacle_features_same_seed_issue_1427_20260524T110529Z`

Both jobs started on `a30` node `auxme-imech172`. Early logs showed repeated node-side NVML
`Driver/library version mismatch` warnings, but both jobs continued past startup. At the time of
this handoff, both jobs had produced base, hardcase, and mixed rollout datasets plus
`training/predictive_model.pt`. The obstacle-feature run also wrote `training/training_summary.json`,
passed its ADE/FDE quality gates, and entered final `scripts/validation/evaluate_predictive_planner.py`.

Observed caveat: proxy evaluation at training checkpoints logged non-fatal `FileNotFoundError`
exceptions for missing `training/proxy_eval/proxy_epoch_*.jsonl` files. The training command
continued and passed the configured quality gates, but final interpretation should check whether
missing proxy metrics should be a hard failure for this comparison.

Verification commands for the launcher branch:

```bash
bash -n SLURM/Auxme/predictive_training_pipeline.sl \
  && bash -n SLURM/Auxme/camera_ready_benchmark.sl
scripts/dev/sbatch_use_max_time.sh --time 12:00:00 --dry-run \
  SLURM/Auxme/predictive_training_pipeline.sl
git diff --check
```

## Follow-Up Boundary

Issue #1167 is not complete until the same-seed training/evaluation comparison reports ADE/FDE and
downstream planner metrics. This patch removes the local pipeline blocker recorded during PR #1400
triage and should be treated as the runnable setup for the campaign, not as benchmark-success
evidence. Issue #1427 owns the actual SLURM submission, metric capture, and recommendation about
whether obstacle features are worth further investment.
