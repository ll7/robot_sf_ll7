# Issue #3482 Release 0.0.2 Provenance Recovery Attempt

This directory records the 2026-07-01 negative provenance recovery attempt for issue #3482. No
valid exact-event provenance artifacts or raw exact-event inputs were found. The missing original
artifacts are:

- `backfill_summary.json`
- `frozen_reconciliation_report.json`
- `reconciliation_tables_0_0_2.jsonl`

The public release bundle supports only the published derived side of the mismatch:

- release `0.0.2` rows: `987`
- published or derived `total_collision_count > 0` rows: `0`

The public release bundle cannot support the exact-event side:

- exact collision outcomes: `241`
- reconciliation violations: `241`

Those exact-event counts require either the original reconciliation artifacts, raw episode-level
records with exact-event fields such as `outcome.collision_event` or
`termination_reason="collision"`, or an equivalent exact-event ledger field.

## Current Disposition

The follow-up disposition packet is
`docs/context/evidence/issue_3482_release_0_0_2_claim_disposition_2026_07_01/`. It explicitly
withdraws release `0.0.2` collision-count-derived claims from paper/dissertation use rather than
validating them. Non-collision release table fields remain available as release-table provenance
with the collision-count caveat.

This supports a `not_planned` / not-recoverable issue closure path after review, not a `completed`
closure path. The empirical collision-count claim was removed from the usable boundary because
source provenance was not recovered.

Valid future supersession paths are:

- recover and promote the original three artifacts with hashes and provenance;
- recover raw exact-event episode records and regenerate the reconciliation deterministically with
  input/output hashes and commands;
- replace this withdrawal with a new reviewed disposition that explicitly documents the promoted
  exact-event provenance.

## Recovery Gate

The recovery gate is:

```bash
uv run python scripts/validation/check_issue_3482_recovery_gate.py --json
```

`--require-close-ready` must fail closed until exact-event provenance is recovered and promoted, or
affected release `0.0.2` collision-count claims are explicitly downgraded or withdrawn. This branch
implements the explicit withdrawal path through the claim disposition packet above.
