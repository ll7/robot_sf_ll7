# Issue #1454 S10 H500 Candidate Evidence

Date: 2026-05-23

This bundle preserves compact, reviewable outputs from the issue #1454 exploratory S10
scenario-horizon h500 campaign with the seven functioning Stage A planner rows plus local
policy-search candidate rows.

## Source Command

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1454_s10_scenario_horizons_h500_candidates.yaml \
  --campaign-id issue1454-s10-h500-candidates
```

The campaign ran at git commit `4941ac48`.

## Durable Raw Archive

The full raw generated campaign tree is preserved as a non-package GitHub artifact release:

- Release:
  <https://github.com/ll7/robot_sf_ll7/releases/tag/artifact/issue1454-s10-h500-candidates-2026-05-23>
- Archive:
  `issue1454-s10-h500-candidates-2026-05-23.tar.zst`
- SHA256:
  `44ec1d4eb89d450eb204398a3807185ce9bdd4aae0eeb5e55af0704fd4a8b0fc`

This is intentionally not a standard Robot SF package release; the tag is namespaced under
`artifact/` so it reads as a durable benchmark payload rather than a versioned software release.

## Result

- Campaign ID: `issue1454-s10-h500-candidates`
- Runtime: `34507.6930` seconds, about `9.6` hours
- Planner rows: `12`
- Successful rows: `12`
- Episodes written: `5760`
- Episodes per row: `480`
- Campaign status: `benchmark_success=true`

Completed rows:

- `goal`
- `social_force`
- `orca`
- `ppo`
- `prediction_planner`
- `socnav_sampling`
- `sacadrl`
- `hybrid_rule_v3_fast_progress`
- `hybrid_rule_v3_fast_progress_static_escape`
- `hybrid_rule_v3_fast_progress_static_escape_continuous`
- `scenario_adaptive_hybrid_orca_v1`
- `scenario_adaptive_hybrid_orca_v2_collision_guard`

The known asset-blocked `socnav_bench` row is intentionally excluded from this exploratory
candidate config. Candidate rows are challenger evidence and should not be mixed into the primary
fixed-h100 versus scenario-horizon verdict without an explicit scope decision.

## Interpretation Notes

The direct outcome table is the primary useful evidence for this run. SNQI diagnostics are
preserved, but the SNQI contract status is `fail`, so SNQI planner ordering should be treated as
diagnostic aggregation rather than a decisive quality ranking.

Top candidate outcome signals from `reports/campaign_table.md`:

| planner | success | collision | near misses | SNQI |
| --- | ---: | ---: | ---: | ---: |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | `0.8771` | `0.0250` | `18.9146` | `-0.0972` |
| `scenario_adaptive_hybrid_orca_v1` | `0.8729` | `0.0333` | `20.7771` | `-0.1037` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | `0.8729` | `0.0333` | `20.7771` | `-0.1037` |
| `hybrid_rule_v3_fast_progress_static_escape` | `0.8646` | `0.0354` | `21.8875` | `-0.1069` |
| `hybrid_rule_v3_fast_progress` | `0.7875` | `0.0292` | `21.9917` | `-0.1160` |

## Included Files

- `campaign_manifest.json`
- `run_meta.json`
- `preflight/validate_config.json`
- `preflight/preview_scenarios.json`
- `reports/campaign_summary.json`
- `reports/campaign_report.md`
- `reports/campaign_table*.{csv,md}`
- `reports/scenario_breakdown.{csv,md}`
- `reports/scenario_family_breakdown.{csv,md}`
- `reports/seed_variability_by_scenario.{csv,json}`
- `reports/seed_episode_rows.csv`
- `reports/statistical_sufficiency.json`
- `reports/snqi_diagnostics.{json,md}`
- `reports/snqi_sensitivity.csv`
- `reports/matrix_summary.{csv,json}`
- `reports/amv_coverage_summary.{json,md}`
- `reports/comparability_matrix.{json,md}`
- `reports/kinematics_*.{csv,md}`
- `runs/*/summary.json`
- `manifest.sha256`

Raw episode JSONL files, full run logs, videos, and the full local `output/` tree are intentionally
not mirrored here. The full raw campaign is available from the artifact release linked above.
