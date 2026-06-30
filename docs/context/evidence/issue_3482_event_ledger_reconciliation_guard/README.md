# Issue #3482 Event-Ledger Reconciliation Guard Evidence

This bundle records the checked-in `EpisodeEventLedger.v1` reconciliation guard fixtures used to
prove the exact/derived collision invariant without committing raw benchmark traces.

## Contents

| File | Source fixture | Expected guard result | Purpose |
| --- | --- | --- | --- |
| `reconciled_fixture_reconciliation.jsonl` | `tests/benchmark/fixtures/event_ledger_reconciliation/reconciled_episodes.jsonl` | pass | Shows a collision event with positive `total_collision_count` and a success episode reconcile cleanly. |
| `violating_fixture_reconciliation.jsonl` | `tests/benchmark/fixtures/event_ledger_reconciliation/violating_exact_collision_zero_count.jsonl` | fail closed | Preserves the regression case: exact collision event plus zero collision count surfaces `exact collision event requires collision metric > 0`. |

## Reproduction

```bash
uv run python scripts/benchmark/export_event_ledger_reconciliation.py \
  --episodes tests/benchmark/fixtures/event_ledger_reconciliation/reconciled_episodes.jsonl \
  --out docs/context/evidence/issue_3482_event_ledger_reconciliation_guard/reconciled_fixture_reconciliation.jsonl \
  --fail-on-violations

uv run python scripts/benchmark/export_event_ledger_reconciliation.py \
  --episodes tests/benchmark/fixtures/event_ledger_reconciliation/violating_exact_collision_zero_count.jsonl \
  --out docs/context/evidence/issue_3482_event_ledger_reconciliation_guard/violating_fixture_reconciliation.jsonl \
  --fail-on-violations
```

The second command is expected to exit `1` after writing the violation row. This is the intended
fail-closed behavior.

## Scope Boundary

This is fixture-level integrity evidence only. It is not a rerun of the frozen `0.0.2` benchmark
traces, does not change paper or dissertation claims, and does not promote any new benchmark result.
