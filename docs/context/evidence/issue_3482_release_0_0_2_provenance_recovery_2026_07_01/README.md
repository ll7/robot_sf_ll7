# Issue #3482 Release 0.0.2 Provenance Recovery Attempt

This directory records the 2026-07-01 negative provenance recovery attempt for issue #3482. No
valid exact-event provenance artifacts or raw exact-event inputs were found.

The missing original artifacts are:

- `backfill_summary.json`
- `frozen_reconciliation_report.json`
- `reconciliation_tables_0_0_2.jsonl`

The public release bundle supports only the published derived side of the mismatch:

- release `0.0.2` rows: `987`
- published or derived `total_collision_count > 0` rows: `0`

The public release bundle cannot support the diagnostic exact-event side:

- exact collision outcomes: `241`
- reconciliation violations: `241`

Those exact-event counts require either the original reconciliation artifacts or raw episode-level
records with exact-event fields such as `outcome.collision_event`,
`termination_reason="collision"`, or an equivalent event-ledger field.

## Current Disposition

The follow-up disposition packet is
`docs/context/evidence/issue_3482_release_0_0_2_claim_disposition_2026_07_01/`.
It withdraws release `0.0.2` collision-count-derived claims for paper/dissertation use rather than
validating them. Non-collision release table fields remain available as release-table provenance
with the collision-count caveat.

This supports a `not_planned` / not-recoverable issue closure path after review, not a `completed`
closure path. The empirical collision-count claim was removed from the usable boundary because the
source provenance was not recovered.
