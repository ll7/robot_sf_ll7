# Issue #3203 Scenario-Horizon Rerun Diagnostic Packet

Related issues: [#3203](https://github.com/ll7/robot_sf_ll7/issues/3203),
[#3266](https://github.com/ll7/robot_sf_ll7/issues/3266),
[#2542](https://github.com/ll7/robot_sf_ll7/issues/2542)

## Status

This packet preserves the fresh bounded #3203 scenario-horizon rerun from 2026-07-01. It is
**diagnostic-only**, not paper-facing Results evidence.

The campaign repaired the stale PPO failure boundary from the 2026-06-20 packet:

- campaign exit code: `0`
- campaign status: `benchmark_success`
- evidence status: `valid`
- total planner rows: `9`
- total episodes: `1296`
- unexpected failed rows: `0`
- fallback/degraded rows counted as success: `0`
- PPO row status: `ok`
- PPO execution mode: `native`
- PPO learned-policy contract: `pass`

The rerun still fails the #3203 promotion gate because the SNQI contract failed:

- SNQI contract status: `fail`
- SNQI rank-alignment Spearman: `-0.19999999999999998`
- SNQI rank-alignment fail threshold: `0.3`
- SNQI outcome separation: `0.29548285685198883`
- SNQI dominant component: `time_penalty`

## Command

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_3203_scenario_horizons_h500_reexport_valid.yaml \
  --output-root <disposable-artifact-root>/benchmarks/issue_3203 \
  --campaign-id issue3203_scenario_horizons_h500_reexport_valid_2026-07-01 \
  --mode run \
  --skip-publication-bundle \
  --log-level INFO
```

The campaign ran at source commit `9cfb5d7df248ca3417d801148a1989d3fd8445f5`, before this branch
was fast-forwarded to current `origin/main`.

## Included Artifacts

- `campaign_manifest.json`: campaign planner/config manifest.
- `run_meta.json`: execution metadata.
- `reports/campaign_summary.json`: structured campaign summary and row statuses.
- `reports/campaign_table.md`: compact planner summary table.
- `reports/scenario_family_breakdown.md`: compact scenario-family breakdown.
- `reports/snqi_diagnostics.json`: structured SNQI contract diagnostics.
- `reports/snqi_diagnostics.md`: readable SNQI diagnostics.
- `checksums.sha256`: checksums for this packet.

## Decision

Do not refresh the dissertation export bundle from this packet. Do not close #3203 as complete.
The next valid promotion path is a bounded follow-up that either fixes the SNQI contract failure or
predeclares a narrower claim boundary that explicitly scopes SNQI out before rerunning.
