# Issue #3482 Release 0.0.2 Provenance Recovery Attempt

This directory records the 2026-07-01 negative provenance recovery attempt for
issue #3482. No valid exact-event provenance artifacts or raw exact-event inputs
were found.

This evidence record does not close #3482. It preserves the blocked boundary so
future recovery work does not regenerate from the public release `0.0.2`
publication bundle alone or repeat the same broad storage searches.

The public release bundle can support only the derived side of the mismatch:

- release `0.0.2` rows: `987`
- published or derived `total_collision_count > 0` rows: `0`

The public release bundle cannot support the exact-event side:

- exact collision outcomes: `241`
- reconciliation violations: `241`

Those exact-event counts require either the original reconciliation artifacts or
raw episode-level records with fields such as `outcome.collision_event`,
`termination_reason="collision"`, or an equivalent exact-event field.

#3482 remains blocked unless one of the following is recovered:

- `backfill_summary.json`
- `frozen_reconciliation_report.json`
- `reconciliation_tables_0_0_2.jsonl`
- raw episode-level records containing exact-event fields

Valid follow-up paths are:

- recover and promote the original three artifacts with hashes and provenance;
- recover raw exact-event episode records and regenerate deterministically with
  input/output hashes and commands;
- explicitly downgrade or withdraw release `0.0.2` collision-count claims if the
  exact-event provenance is permanently unavailable.

Recommended next actions:

- verify the host identity and scratch retention status for `imech156` /
  `imech156-u` through an authoritative source before attempting SSH recovery;
- audit paper-facing and dissertation-facing release `0.0.2` result tables that
  rely on `total_collision_count` or collision-count-derived claims;
- keep #3482 blocked unless exact-event provenance is recovered, or close it only
  as not recoverable after the affected release `0.0.2` collision-count claims
  are explicitly downgraded or withdrawn.
