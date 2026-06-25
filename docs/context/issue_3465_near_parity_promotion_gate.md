# Issue #3465 — near-parity promotion-gate decision (increment)

**Status:** diagnostic / decision layer. Provides the formal claim-boundary verdict #2540 deferred.

## What this is

`robot_sf/benchmark/near_parity_promotion_gate.py` is the pure decision layer for #3465. It turns the
paired gate-enabled-vs-disabled comparison of the topology near-parity selector into the issue's
verdict — `diagnostic`, `revise`, `stop`, or `eligible_for_promotion` — fail-closed about
fallback/degraded execution. It mirrors the accepted decision-layer pattern (#3484, #3558, #3557).

## Decision (`near_parity_promotion_gate.v1`)

`classify_near_parity_promotion(NearParityComparison, thresholds)` decides (first match wins):

1. corrective work (#3463) not complete → `diagnostic` (formal test not runnable);
2. improvement relies on fallback/degraded rows → `revise` (fail-closed);
3. measurable safety/efficiency **regression** → `stop`;
4. measurable, **paired-significant** native improvement → `eligible_for_promotion` (only promote);
5. otherwise → `revise`.

## Scope boundary

Pure and side-effect free. Freezing the paired benchmark config and **running** the gate-enabled/
disabled arms over common seeds needs the #3463 corrective work plus cluster execution and is the
deliberate deferred follow-up; this layer turns those results into the verdict.

## Tests

`tests/benchmark/test_near_parity_promotion_gate.py` (6 tests): each verdict branch — diagnostic
(corrective incomplete), revise (fallback-reliant / non-significant / no-difference), stop
(regression), and eligible_for_promotion (significant native improvement).

## Related

- Predecessor: #2540 (`revise`). Corrective implementation: #3463.
