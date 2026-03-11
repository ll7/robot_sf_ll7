# BR-07 Evening Run: Predictive Planner Refresh

This guide prepares and launches issue #577 (predictive-planner data refresh + retraining)
after BR-06 model finalization.

## Preconditions
- Repository is on a branch containing BR-07 prep updates.
- `uv sync --all-extras` completed.
- Optional: authenticate W&B if logging externally.

## One-Copy Launch Bundle
```bash
set -euo pipefail

export SCENARIO_MATRIX="configs/scenarios/classic_interactions.yaml"
export HARD_SEEDS="configs/benchmarks/predictive_hard_seeds_v1.yaml"
export PLANNER_GRID="configs/benchmarks/predictive_sweep_planner_grid_v1.yaml"
export TS=$(date +%Y%m%d_%H%M%S)

export BASE_DATASET="output/tmp/predictive_planner/datasets/predictive_rollouts_full_v2_${TS}.npz"
export HARD_DATASET="output/tmp/predictive_planner/datasets/predictive_rollouts_hardcase_v2_${TS}.npz"
export MIXED_DATASET="output/tmp/predictive_planner/datasets/predictive_rollouts_mixed_v2_${TS}.npz"
export TRAIN_DIR="output/tmp/predictive_planner/training/predictive_proxy_selected_v2_${TS}"
export CHECKPOINT="$TRAIN_DIR/predictive_model.pt"

echo "[BR-07] Collect base dataset -> $BASE_DATASET"
uv run python scripts/training/collect_predictive_planner_data.py \
  --episodes 200 \
  --output "$BASE_DATASET"

echo "[BR-07] Collect hardcase dataset -> $HARD_DATASET"
uv run python scripts/training/collect_predictive_hardcase_data.py \
  --scenario-matrix "$SCENARIO_MATRIX" \
  --seed-manifest "$HARD_SEEDS" \
  --output "$HARD_DATASET"

echo "[BR-07] Build mixed dataset -> $MIXED_DATASET"
uv run python scripts/training/build_predictive_mixed_dataset.py \
  --base-dataset "$BASE_DATASET" \
  --hardcase-dataset "$HARD_DATASET" \
  --output "$MIXED_DATASET"

echo "[BR-07] Train predictive model with proxy selection -> $TRAIN_DIR"
uv run python scripts/training/train_predictive_planner.py \
  --dataset "$MIXED_DATASET" \
  --output-dir "$TRAIN_DIR" \
  --select-by-proxy \
  --proxy-scenario-matrix "$SCENARIO_MATRIX" \
  --proxy-seed-manifest "$HARD_SEEDS"

echo "[BR-07] Evaluate checkpoint -> $CHECKPOINT"
uv run python scripts/validation/evaluate_predictive_planner.py \
  --checkpoint "$CHECKPOINT" \
  --scenario-matrix "$SCENARIO_MATRIX"

uv run python scripts/validation/run_predictive_hard_seed_diagnostics.py \
  --scenario-matrix "$SCENARIO_MATRIX" \
  --seed-manifest "$HARD_SEEDS" \
  --checkpoint "$CHECKPOINT"

uv run python scripts/validation/run_predictive_success_campaign.py \
  --checkpoints "$CHECKPOINT" \
  --scenario-matrix "$SCENARIO_MATRIX" \
  --hard-seed-manifest "$HARD_SEEDS" \
  --planner-grid "$PLANNER_GRID"

echo "[BR-07] Benchmark-facing policy analysis (videos enabled)"
uv run python scripts/tools/policy_analysis_run.py \
  --scenario "$SCENARIO_MATRIX" \
  --policy prediction_planner \
  --algo-config configs/algos/prediction_planner_camera_ready.yaml \
  --seed-set eval \
  --max-seeds 3 \
  --videos

echo "[BR-07] Done. Checkpoint: $CHECKPOINT"
```

## Required Outputs
- `output/tmp/predictive_planner/datasets/predictive_rollouts_full_v2_<ts>.npz`
- `output/tmp/predictive_planner/datasets/predictive_rollouts_hardcase_v2_<ts>.npz`
- `output/tmp/predictive_planner/datasets/predictive_rollouts_mixed_v2_<ts>.npz`
- `output/tmp/predictive_planner/training/predictive_proxy_selected_v2_<ts>/predictive_model.pt`
- `output/tmp/predictive_planner/training/predictive_proxy_selected_v2_<ts>/training_summary.json`
- Validation outputs under `output/validation/...`
- Policy-analysis outputs under `output/benchmarks/<timestamp>_policy_analysis_prediction_planner/`
- Videos under `output/recordings/<timestamp>_policy_analysis_prediction_planner/`

## Stop Conditions
Stop and open a follow-up issue if any of the following happens:
- checkpoint file missing after training,
- hard-seed diagnostics report any contradiction records,
- policy-analysis output contains success/collision contradictions,
- proxy selection fails to produce selected checkpoint metadata.
