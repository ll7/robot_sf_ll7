# Issue #1454 Stage A Fixed-H100 Evidence

Date: 2026-05-22

This bundle preserves the compact, reviewable outputs from the issue #1454 Stage A S10 fixed-h100
broader-baseline campaign.

## Source Command

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1454_s10_fixed_h100_broader_baselines.yaml \
  --campaign-id issue1454-s10-fixed-h100
```

The campaign ran at git commit `17b179007e6292dd0365c53cff719cccea1276a9`.

## Result

- Campaign ID: `issue1454-s10-fixed-h100`
- Runtime: `2722.0126` seconds
- Episodes written: `3360`
- Planner rows: `8`
- Completed rows: `7`
- Failed rows: `1`
- Campaign status: `benchmark_success=false`

The completed rows each wrote `480` episodes:

- `goal`
- `social_force`
- `orca`
- `ppo`
- `prediction_planner`
- `socnav_sampling`
- `sacadrl`

The `socnav_bench` row failed closed with `0` episodes because local SocNavBench assets were
missing. The run summary points to `docs/socnav_assets_setup.md` and
`uv run python scripts/tools/prepare_socnav_assets.py`. This is the same dependency boundary tracked
by issue #562 and `docs/context/issue_562_socnav_bench_reentry.md`.

## Included Files

- `reports/campaign_summary.json`
- `reports/campaign_report.md`
- `reports/campaign_analysis.json`
- `reports/campaign_analysis.md`
- `reports/scenario_difficulty_analysis.json`
- `reports/scenario_difficulty_analysis.md`
- `reports/scenario_breakdown.csv`
- `reports/scenario_family_breakdown.csv`
- `reports/statistical_sufficiency.json`
- `reports/seed_variability_by_scenario.json`
- `reports/snqi_diagnostics.json`
- `reports/snqi_diagnostics.md`
- `reports/may4_fixed_h100_vs_issue1454_s10_comparison.json`
- `reports/may4_fixed_h100_vs_issue1454_s10_comparison.md`
- `runs/*/summary.json`
- `manifest.sha256`

Raw episode JSONL, videos, and full `output/` campaign contents are intentionally not mirrored here.
They remain worktree-local and reproducible from the tracked config, seed schedule, commit, and
commands.

## Gate Decision

Stage B is a no-go from this Stage A run. The issue #1454 plan requires Stage A to have no
unresolved campaign-level failure before the scenario-horizon h500 comparison is interpreted. Since
`socnav_bench` failed closed, the correct next step is to repair or explicitly scope out the
SocNavBench row before rerunning Stage A or comparing fixed-h100 against h500.

The durable gate note is `docs/context/issue_1454_stage_a_gate_2026-05-22.md`.
