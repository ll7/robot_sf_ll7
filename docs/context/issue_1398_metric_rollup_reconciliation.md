# Issue #1398 Metric Rollup Reconciliation

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1398>

## Scope

Issue #1398 followed the 2026-05-20 SLURM batch for #1344 and #1354. The batch exposed two
separate interpretation problems:

- collision outcomes could be true while sampled collision metrics stayed non-positive,
- analyzer SNQI row-vs-episode checks could disagree with campaign report rows.

The collision-source problem is handled by the #1344/#1399 map-runner fix, which floors collision
metrics from exact typed environment collision flags before outcome-integrity validation.

This note records the analyzer-side SNQI reconciliation.

## Decision

`scripts/tools/analyze_camera_ready_campaign.py` now computes analyzer SNQI episode means through
the same canonical helper used by the camera-ready campaign writer:

- `robot_sf.benchmark.utils.episode_metric_value(record, "snqi")`

This matters because episode artifacts can carry SNQI either inside `metrics.snqi` or as a top-level
episode field. The campaign writer already accepts both shapes when building `planner_rows`.
The analyzer previously read only `metrics.snqi`, so it could report a row-vs-episode SNQI mismatch
even when the row was consistent with the canonical episode metric contract.

## Interpretation Boundary

After this change:

- `snqi_mean mismatch` in `campaign_analysis` means the planner row and canonical episode metric
  values really disagree beyond tolerance,
- absence of that mismatch only proves row/episode accounting consistency, not SNQI claim quality,
- `snqi_contract_status=fail` or `warn` remains the claim-scope signal for paper-facing SNQI use.

For #1344, the paired primary AMV report still remains non-paper-facing because AMV coverage is
incomplete and the SNQI contract fails. For #1354, cross-kinematics execution proof should keep
compatibility and SNQI-contract interpretation separate: a campaign can run successfully while still
being unsuitable for final paper-facing SNQI claims.

## Validation

```bash
uv run pytest tests/tools/test_analyze_camera_ready_campaign.py::test_analyze_campaign_uses_canonical_snqi_episode_metric_value -q
uv run pytest tests/tools/test_analyze_camera_ready_campaign.py tests/unit/benchmark/test_utils_episode_helpers.py -q

uv sync --all-extras

uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1344_paired_nominal_v1_primary.yaml \
  --mode run \
  --output-root output/benchmarks/issue_1398 \
  --campaign-id issue_1398_nominal_rollup_smoke_synced \
  --log-level WARNING

uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1344_paired_stress_primary.yaml \
  --mode run \
  --output-root output/benchmarks/issue_1398 \
  --campaign-id issue_1398_stress_rollup_smoke_synced \
  --log-level WARNING

uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1398/issue_1398_nominal_rollup_smoke_synced

uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1398/issue_1398_stress_rollup_smoke_synced
```

Results:

- targeted canonical SNQI test: passed,
- analyzer/helper test slice: `48 passed`,
- nominal #1344 bounded rerun: `benchmark_success=true`, `3/3` rows, `36` episodes, no warnings,
- stress #1344 bounded rerun: `benchmark_success=true`, `3/3` rows, `432` episodes, no warnings,
- analyzer findings for both reruns: `[]`.

The first local nominal attempt before `uv sync --all-extras` failed closed because the fresh
worktree was missing the ORCA `rvo2` extra. After syncing all extras, ORCA ran natively and the
bounded reruns passed.
