# #9861: fail closed on invalid or unmeasured reset clearance

## Goal and scope

For the corrected 0.0.8 producer, a reset overlap or unavailable reset-clearance
measurement invalidates the row regardless of route completion. Preserve the
observed outcome in raw records and the event ledger. Exclude invalid rows from
nominal rates and SNQI-v2 calibration. Keep historical 0.0.7 artifacts unchanged.

## Evidence and decision

- Issue #9861 records two exact counterexamples in `build_spawn_validity`:
  completed route with a reset overlap, and unavailable reset clearance.
- The comparison baseline is the 0.0.7 rule under `spawn_validity.v1`; the
  correction emits `spawn_validity.v2`. A changed outcome is explained by that
  versioned producer and must still pass the paired 0.0.7/0.0.8 release review.
- An observed route completion remains an event fact, but an invalid start never
  counts as planner success. Unknown clearance is reported separately from a
  measured overlap. Route-end respawn attribution retains its existing
  completed-route exception.
- Fallback, degraded, partial, and invalid rows are diagnostic, not nominal
  benchmark evidence. The pinned release matrix and Chapter 7 claim review are
  outside this PR and remain release gates.

## Implementation and proof

1. Emit versioned reset validity and separate reasons; propagate invalidity to
   aggregate, camera-ready, parquet, and seed-rate consumers. Those consumers
   also inspect v2 reset telemetry directly, so a forged valid flag cannot
   admit an overlapping, unavailable, or incomplete reset block; legacy v1
   rows retain their historical diagnostic comparison behavior.
2. Reconcile `goal_reached && invalid_run` only for a versioned invalid-start
   provenance; continue rejecting unexplained contradictory ledgers.
3. Reject invalid and forged-valid unmeasured resets before SNQI-v2 score or
   calibration. Assert completed-route overlap and unknown-clearance exclusions.
4. Run focused benchmark, SNQI-v2, ledger, aggregate-golden, and runner tests;
   Ruff, formatting, diff, documentation integrity, and final PR readiness.

## Recovery and handoff

The isolated branch `issue-9861-spawn-row-gate` owns this packet. Its PR must
receive independent review at the final head. The PR is implementation evidence
only; 0.0.8 admission still requires the pinned-matrix preflight, candidate
receipt, paired release comparison, and author gates in #9668.
