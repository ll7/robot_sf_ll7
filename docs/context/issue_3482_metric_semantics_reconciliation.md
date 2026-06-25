# Issue #3482 — Metric-semantics reconciliation table (increment)

**Status:** integrity infra. The core `EpisodeEventLedger.v1` and its reconciliation guard
(exact collision ⇒ collision metric > 0; goal/invalid_run mutual exclusion; metric-definition
presence) already exist and are property-tested in `robot_sf/benchmark/event_ledger.py` +
`tests/benchmark/test_event_ledger.py`. This increment adds the remaining DoD artifact.

## What this is

`robot_sf/benchmark/event_ledger_reconciliation.py` builds the **metric-semantics
reconciliation table** for one episode ledger: for each reported metric field it records the
producer, sampling level, kind (`exact_or_sampled` | `derived`), representative downstream
consumers, and surfaces the overall audit result + any reconciliation violations. It is a pure
function over an `EpisodeEventLedger.v1` payload, so every table in the empirical chapter gets a
single, testable provenance row — closing the "easiest examiner attack" (an exact collision flag
silently coexisting with a zero collision count).

## Output (`event_ledger_reconciliation.v1`)

`build_metric_semantics_table(ledger)` →

- `rows`: one per metric field (`field`, `level`, `kind`, `producer`, `representative_consumers`)
  for `collision_count`, `near_miss`, `clearance_breach`, `ttc_breach`, `oscillation`;
- `audit_result`, `reconciles`, and `violations` from `reconcile_event_ledger`.

`representative_consumers` is intentionally conservative (audit orientation, not an exhaustive
dependency graph).

## Scope boundary

Pure and side-effect free. The remaining DoD item — re-running the frozen `0.0.2` traces through
the corrected analyzer for a before/after numerical diff — needs the archived artifacts and is a
deliberate follow-up.

## Tests

`tests/benchmark/test_event_ledger_reconciliation.py`: one row per reported field with valid
kind/producer, collision field is exact/sampled with a named producer, a clean ledger reconciles
(audit pass), and an exact-collision-vs-zero-metric disagreement surfaces as a violation.
