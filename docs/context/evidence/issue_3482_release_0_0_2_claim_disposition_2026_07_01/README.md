# Issue #3482 Release 0.0.2 Collision-Count Claim Disposition

This packet records the closure-path decision for the release `0.0.2` collision-count
integrity blocker after the 2026-07-01 `imech156-u` recovery pass.

The original reconciliation artifacts were not found:

- `backfill_summary.json`
- `frozen_reconciliation_report.json`
- `reconciliation_tables_0_0_2.jsonl`

No raw release `0.0.2` episode-level records with exact-event fields were found either. The
public release bundle remains useful for release-table provenance, but it only supports the
published derived side of the mismatch: `987` rows and zero positive published
`total_collision_count` rows. It does not prove the diagnostic issue-comment counts of `241`
exact collision outcomes or `241` exact/derived violations.

## Claim Disposition

- Exact collision incidence for release `0.0.2`: bounded diagnostic issue-comment evidence only,
  not release-bundle evidence.
- Release `0.0.2` `total_collision_count` and collision-count-derived claims: withdrawn for
  paper/dissertation use unless a future promoted reconciliation bundle supersedes this packet.
- Non-collision release table fields: unchanged by this packet, but they must not be cited as
  collision-count evidence.

## Affected Tables

- `tab_release_failure_count_slices`: collision-count/failure-count interpretations withdrawn;
  use only for non-collision context unless superseded.
- `tab_results_overview`: collision-count columns withdrawn; non-collision fields remain release
  provenance only.
- `tab_robot_sf_release_planner_results`: collision-count columns withdrawn; non-collision fields
  remain release provenance only.

See `manifest.json` for the machine-readable disposition.
