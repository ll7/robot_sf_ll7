# Quickstart — Social Navigation Benchmark

This guide demonstrates the end-to-end benchmark workflow (WHAT you do, not internal HOW):

## 1. Validate Scenario Matrix
Provide a matrix file (YAML/JSON) defining ≥12 core scenarios.
```
robot_sf_bench validate-config --matrix configs/baselines/scenario_matrix.yaml
```

## 2. Run Episodes (Generate JSONL)
```
robot_sf_bench run \
  --matrix configs/baselines/scenario_matrix.yaml \
  --output results/episodes.jsonl \
  --algo social_force --algo ppo --algo random \
  --workers 4 --seed 123 --resume
```
Result: `results/episodes.jsonl` + `episodes.jsonl.manifest.json`.

## 3. Compute Baseline Stats & SNQI Weights
```
robot_sf_bench baseline \
  --episodes results/episodes.jsonl \
  --output results/baseline_stats.json

robot_sf_snqi recompute \
  --baseline-stats results/baseline_stats.json \
  --out weights/snqi_weights_v1.json
```

## 4. Aggregate Metrics with Confidence Intervals
```
robot_sf_bench aggregate \
  --in results/episodes.jsonl \
  --out results/summary_ci.json \
  --bootstrap-samples 1000 --bootstrap-confidence 0.95 --bootstrap-seed 42
```

## 5. Generate Figures & Tables
```
python scripts/generate_figures.py \
  --episodes results/episodes.jsonl \
  --table-summary results/summary_ci.json \
  --table-include-ci --table-tex \
  --out-dir docs/figures/episodes_run_v1
```
Artifacts: Pareto plots, distribution plots, force-field figures, scenario thumbnails, baseline table (Markdown + LaTeX), SNQI ablation outputs.

## 6. SNQI Ablation (Sensitivity)
```
robot_sf_snqi ablation --episodes results/episodes.jsonl --summary-out results/snqi_ablation.json
```

## 7. Resume Behavior (Incremental Additions)
Re-run step 2 with new algorithms or extended repetitions; existing episodes are skipped (manifest-driven) and only new episodes appended.

## 8. Reproducibility Check
Run steps 2–5 with a different seed (e.g., 456) and compare aggregated metrics—expect differences within bootstrap CIs.

## Outputs Summary
- episodes.jsonl (raw per-episode lines)
- episodes.jsonl.manifest.json (resume index)
- baseline_stats.json (normalization stats)
- snqi_weights_v1.json (weights artifact)
- summary_ci.json (aggregated metrics + optional CIs)
- docs/figures/... (visual assets & tables)
- snqi_ablation.json (component influence)

## Next Steps
- Add optional ORCA baseline once licensing cleared.
- Expand scenario matrix with real-data-calibrated variant.

---
All commands are illustrative; final CLI flag names must match published docs.
