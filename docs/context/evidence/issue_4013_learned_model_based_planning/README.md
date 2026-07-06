# Issue #4013 Learned-Prediction MPC Evidence

This directory records the intended durable evidence shape for issue #4013. The
comparison remains diagnostic until real paired episode JSONL files exist for all
three required arms: `learned_prediction_mpc`, `cv_prediction_mpc`, and one
model-free baseline.

Use:

```bash
uv run python scripts/analysis/compare_model_based_planning_issue_4013.py \
  --config configs/analysis/issue_4013_model_based_planning_comparison.yaml
```

The report fails closed unless matched scenario/seed rows exist for all three
arms, fallback/degraded rows are excluded, and the claim boundary remains
diagnostic-only. It does not run a full benchmark campaign, train a predictor, or
make paper/dissertation claims.

## Trained short-horizon predictor (diagnostic)

The predictor can now be trained on CPU so the model-based arm loads real learned
weights instead of the zero-initialized untrained smoke model:

```bash
uv run python scripts/training/train_learned_short_horizon_predictor_issue_4013.py \
  --config configs/training/learned_short_horizon_predictor_issue_4013_smoke.yaml
```

This writes a checkpoint plus manifest and metrics under
`output/models/issue_4013/short_horizon_predictor/` (git-ignored). A copy of the
manifest and metrics from a local run is promoted here as durable evidence:

- `training_manifest.v1.json` — schema, architecture, trainer/predictor config, claim boundary.
- `training_metrics.v1.json` — initial/final training loss.

Observed local run (seed 4013, 512 samples, 400 epochs, CPU): training loss fell
from `0.0987` to `0.00044` and the resulting checkpoint loads into
`LearnedShortHorizonPedestrianPredictor` with `evidence_tier=checkpoint_loaded`
(not `diagnostic_untrained_smoke`).

Claim boundary: the training task is a seeded synthetic robot-repulsion
learnability probe, **not** real ETH/UCY pedestrian data. The checkpoint is
`smoke evidence` that the predictor trains and loads without fallback; it is not
benchmark, navigation-quality, or paper/dissertation evidence. The still-open
remainder is running the paired smoke scenario for all three arms and feeding the
episode JSONL into the comparison report above.
