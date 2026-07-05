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
