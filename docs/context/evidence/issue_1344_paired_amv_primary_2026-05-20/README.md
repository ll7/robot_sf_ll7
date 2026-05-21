# Issue #1344 Paired AMV Primary Evidence

This bundle preserves compact, reviewable evidence for the issue #1344 paired nominal/stress AMV
primary-row protocol. Raw episode JSONL and other full campaign output remain in the git-ignored
`output/` tree.

## Campaigns

Both campaigns ran from commit `c16ae67b5fe2c605476152113d43e569828958a7` with the S3 `eval` seed
set `[111, 112, 113]`, differential-drive kinematics, and primary/core planners only:
`goal`, `social_force`, and `orca`.

| Surface | Campaign ID | Scenario matrix | Matrix hash | Episodes | Successful rows | Warnings |
| --- | --- | --- | --- | ---: | ---: | --- |
| nominal | `issue_1344_nominal_primary_final` | `configs/scenarios/nominal_v1.yaml` | `73acddfd12cf` | 36 | 3/3 | none |
| stress | `issue_1344_stress_primary_final` | `configs/scenarios/classic_interactions_francis2023.yaml` | `8ac8ab9387f4` | 432 | 3/3 | none |

Both campaigns report `benchmark_success=true`, meaning the configured primary rows completed and
passed fail-closed campaign execution. They also report `amv_coverage_status=warn` and
`snqi_contract_status=fail`. The AMV warning is not partial coverage: the copied coverage summaries
show `Observed = -` for every required AMV dimension, so all required AMV dimension values are
missing from the source scenario metadata. The campaigns are `paper_facing=false`; treat them as
protocol/outlook evidence, not paper-claim evidence.

## Files

- `nominal_campaign_summary.json`: compact nominal campaign summary.
- `nominal_campaign_table.md`: nominal planner table.
- `nominal_amv_coverage_summary.md`: nominal AMV coverage summary.
- `stress_campaign_summary.json`: compact stress campaign summary.
- `stress_campaign_table.md`: stress planner table.
- `stress_amv_coverage_summary.md`: stress AMV coverage summary.
- `manifest.sha256`: checksums for the copied evidence files.

## Reproduction

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1344_paired_nominal_v1_primary.yaml \
  --mode preflight \
  --output-root output/benchmarks/issue_1344 \
  --campaign-id issue_1344_nominal_primary_final_preflight

uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1344_paired_stress_primary.yaml \
  --mode preflight \
  --output-root output/benchmarks/issue_1344 \
  --campaign-id issue_1344_stress_primary_final_preflight

uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1344_paired_nominal_v1_primary.yaml \
  --mode run \
  --output-root output/benchmarks/issue_1344 \
  --campaign-id issue_1344_nominal_primary_final \
  --log-level INFO

uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1344_paired_stress_primary.yaml \
  --mode run \
  --output-root output/benchmarks/issue_1344 \
  --campaign-id issue_1344_stress_primary_final \
  --log-level INFO
```
