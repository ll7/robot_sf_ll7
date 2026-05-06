# Issue 1023 Candidate-Augmented Preflight Evidence

Date: 2026-05-06

This compact evidence bundle records the local preflight for the h500 scenario-horizon benchmark
after adding the two experimental candidate planners:

- `scenario_adaptive_hybrid_orca_v1`
- `hybrid_rule_v3_fast_progress_static_escape`

Command:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --mode preflight \
  --campaign-id issue1023_scenario_horizons_candidates_preflight_2026-05-06 \
  --log-level INFO
```

Summary:

- Scenarios: 48.
- Planners: 9.
- Horizon mode: `scenario_horizons`.
- Horizon source: `configs/policy_search/scenario_horizons_h500.yaml`.
- Horizon range: 102-600 steps.
- Horizon status counts: 45 `recommended`, 3 `planner_blocked`.

Tracked files:

- `preflight/validate_config.json`: config validation summary.
- `reports/matrix_summary.csv`: planner matrix summary with the two candidate rows.

The full preflight output remains in ignored local `output/` and is not required for review.
Outcome evidence for the same 9-planner matrix is tracked separately in
`docs/context/evidence/issue_1023_candidate_augmented_local_full_2026-05-06/`.
