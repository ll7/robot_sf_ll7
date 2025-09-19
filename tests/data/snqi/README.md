# SNQI Synthetic Fixture Dataset

Deterministic small dataset (6 episodes) used for:
- Fast unit/integration tests of optimization scripts
- Future drift detection (regression) tests
- Placeholder for upcoming bootstrap CI logic

Design choices:
- Two algos (A, B) to exercise ranking stability path (Spearman branch)
- Variation across safety/time metrics to produce non-trivial variance
- Baseline stats chosen so all raw metrics fall within [0, p95] (no saturation except potential jerk_mean upper values approaching p95)

Files:
- `episodes_small.jsonl` — JSON Lines episodes with metrics
- `baseline_stats.json` — median/p95 for normalization-driven metrics

Reproducibility:
- Content is static; modify only with corresponding snapshot test updates.
