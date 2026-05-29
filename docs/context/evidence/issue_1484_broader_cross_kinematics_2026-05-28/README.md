# Issue #1484 Broader Cross-Kinematics Evidence 2026-05-28

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1484>
- <https://github.com/ll7/robot_sf_ll7/issues/1353>
- <https://github.com/ll7/robot_sf_ll7/issues/1354>

## Scope

This directory preserves compact, reviewable smoke/probe evidence for the #1484 broader
cross-kinematics matrix. Raw run bundles remain under ignored `output/`.

## Campaign

- Job: `12658`
- Result: `COMPLETED 0:0`
- Partition: `a30`
- Elapsed: `00:02:04`
- Commit: `f35fb3b5d47c217594a7d6e923b14bc22772ab5f`
- Config: `configs/benchmarks/issue_1484_broader_cross_kinematics.yaml`
- Campaign id:
  `issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441`

## Key Outcomes

- `total_runs=21`
- `successful_runs=21`
- `total_episodes=21`
- `benchmark_success=true`
- `benchmark_success_basis=core`
- `core_successful_runs=9`
- `core_total_runs=9`
- `amv_coverage_status=pass`
- `snqi_contract_status=warn`
- Analyzer findings: none

The executed matrix is seven planners across three kinematics modes:

- core: `goal`, `social_force`, `orca`
- experimental: `ppo`, `prediction_planner`, `sacadrl`, `socnav_sampling`
- kinematics: `differential_drive`, `bicycle_drive`, `holonomic`

## Interpretation Boundary

This is a compatibility smoke/probe, not campaign-sized #1484 evidence. The executed surface is one
scenario, one seed, three kinematics modes, seven planner rows, and horizon `80`, producing 21 short
episodes and an observed SLURM runtime of about two minutes. It proves that the broader matrix
executes and reports cleanly; it does not close the original broader-campaign objective without a
larger scenario/seed/runtime surface or an explicit maintainer scope decision.

## Caveats

- This is broader compatibility smoke/probe evidence, not paper-facing evidence by itself.
- The run is a one-scenario, one-seed smoke-style matrix.
- `ppo` on `holonomic` reported learned-policy contract `warn`.
- `socnav_sampling` is still an experimental SocNav-family row and logged heuristic-fallback
  warnings during runtime.
- `socnav_bench` is intentionally excluded in the config-level compatibility manifest because
  Issue #1353 preserved it as `not_available` due to missing SocNavBench control-pipeline assets.

## Preserved Files

- `campaign_manifest.json`
- `reports/campaign_summary.json`
- `reports/campaign_table.csv`
- `reports/campaign_table.md`
- `reports/campaign_analysis.json`
- `reports/campaign_analysis.md`
- `reports/kinematics_parity_table.csv`
- `reports/kinematics_parity_table.md`
- `reports/kinematics_skipped_combinations.csv`
- `reports/kinematics_skipped_combinations.md`
- `reports/matrix_summary.json`
- `reports/matrix_summary.csv`
- `reports/amv_coverage_summary.json`
- `reports/amv_coverage_summary.md`
- `reports/snqi_diagnostics.json`
- `reports/snqi_diagnostics.md`
- other compact report tables from the campaign `reports/` directory

Checksums are recorded in `manifest.sha256`.
