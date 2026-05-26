# Issue #1353 Benchmark Evidence 2026-05-26

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1353>
- <https://github.com/ll7/robot_sf_ll7/issues/1354>
- <https://github.com/ll7/robot_sf_ll7/issues/1517>

## Scope

This directory preserves compact, reviewable benchmark evidence from the 2026-05-26 SLURM sweep
after the current `main` checkout included core-anchored campaign success semantics and the local
virtualenv was restored with the ORCA `rvo2` extra.

Raw campaign outputs remain under ignored `output/benchmarks/...`. This directory intentionally
keeps only summaries, tables, analysis reports, and checksums.

## Campaigns

| Surface | Job | Config | Campaign root | Result |
| --- | ---: | --- | --- | --- |
| #1353 nominal | `12626` | `configs/benchmarks/issue_1353_paired_nominal_v1_broader_baselines.yaml` | `output/benchmarks/issue_1353/issue_1353_paired_nominal_v1_broader_baselines_issue1353-nominal-main-rvo2-20260526_20260526_062603` | `COMPLETED 0:0`, core `3/3`, all-row `7/8` |
| #1353 stress | `12625` | `configs/benchmarks/issue_1353_paired_stress_broader_baselines.yaml` | `output/benchmarks/issue_1353/issue_1353_paired_stress_broader_baselines_issue1353-stress-main-rvo2-20260526_20260526_062603` | `COMPLETED 0:0`, core `3/3`, all-row `7/8` |
| #1354 compact cross-kinematics | `12624` | `configs/benchmarks/paper_cross_kinematics_v1.yaml` | `output/benchmarks/issue_1354_20260526/paper_cross_kinematics_v1_issue1354-paper-compact-main-rvo2-20260526_20260526_062604` | `COMPLETED 0:0`, core `9/9` |

## Key Outcomes

- #1354 has a clean compact core cross-kinematics run: `9/9` successful rows,
  `benchmark_success=true`, `amv_coverage_status=pass`, and `snqi_contract_status=warn`.
- #1353 nominal completed with `benchmark_success=true` on core rows:
  `total_runs=8`, `successful_runs=7`, `total_episodes=84`, `core_successful_runs=3`,
  `core_total_runs=3`.
- #1353 stress completed with `benchmark_success=true` on core rows:
  `total_runs=8`, `successful_runs=7`, `total_episodes=1008`, `core_successful_runs=3`,
  `core_total_runs=3`.
- #1353 nominal and stress remain AMV-coverage gap evidence, not clean AMV-coverage success:
  both `amv_coverage_summary.*` files report `status=warn` with missing required AMV dimensions.
- `socnav_bench` is explicitly `not_available` in both #1353 surfaces because SocNavBench
  control-pipeline assets are missing. It is caveated evidence, not a successful planner row.
- Analyzer findings were empty for all three fresh campaign roots.
- #1353 stress still has `snqi_contract_status=fail`, matching the prior #1344 caution that SNQI
  should not be promoted into paper-facing claims without a separate claim-scope decision.

## Preserved Files

Each surface keeps:

- `campaign_summary.json`
- `campaign_table.csv`
- `campaign_table.md`
- `campaign_analysis.json`
- `campaign_analysis.md`
- `scenario_difficulty_analysis.json`
- `scenario_difficulty_analysis.md`
- `matrix_summary.json`
- `matrix_summary.csv`
- `amv_coverage_summary.json`
- `amv_coverage_summary.md`
- `snqi_diagnostics.json`
- `snqi_diagnostics.md`

Checksums are recorded in `manifest.sha256`.

## Validation Commands

```bash
sacct -j 12624,12625,12626 \
  --format=JobID,JobName%35,Partition,State,ExitCode,Elapsed,Timelimit,NodeList%30

uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1353/issue_1353_paired_nominal_v1_broader_baselines_issue1353-nominal-main-rvo2-20260526_20260526_062603

uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1353/issue_1353_paired_stress_broader_baselines_issue1353-stress-main-rvo2-20260526_20260526_062603

uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1354_20260526/paper_cross_kinematics_v1_issue1354-paper-compact-main-rvo2-20260526_20260526_062604
```
