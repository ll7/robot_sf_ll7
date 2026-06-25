# Issue #3483 — Trace-level safety-predicate producers (oscillatory, first increment)

**Status:** diagnostic / proxy. **Evidence grade:** idea-level; predicate definitions are
versioned modeling choices, labeled diagnostic until real-world label validation
(externally blocked, #3278). Do **not** report predicate rates in the manuscript until
fields are emitted and validated.

## What this is

`robot_sf/benchmark/safety_predicates.py` provides in-sim producers for the trace-level
safety predicates the thesis motivates, so their fields become reproducible and
fixture-backed instead of simulation-only assumptions.

This increment implements the **oscillatory local-control** predicate (the most
self-contained of the three). Its boolean output is compatible with the existing
`SurrogateEvents.oscillation` slot in `robot_sf/benchmark/event_ledger.py`; the detailed
field record is intended for the ledger `surrogate_events` block.

## Producer (`safety_predicate.oscillatory_control.v1`)

`oscillatory_control_predicate(positions, headings, linear_velocities, *, dt, …)` emits:

- `heading_rate_sign_changes`, `linear_velocity_sign_changes`, `command_source_changes`
- `net_progress_m`, `path_length_m`, `progress_ratio`
- `mean_abs_jerk`, `n_steps`

Diagnostic classification: `oscillation = (heading_rate_sign_changes ≥ T_h) and
(progress_ratio ≤ T_p)` with defaults `T_h = 4`, `T_p = 0.5`. Thresholds are explicit and
overridable, and the raw fields are always emitted so a different threshold can be applied
downstream without recomputation.

## Scope boundary

Pure and side-effect free — changes no runtime/benchmark behavior. Deferred follow-ups:

- the **late-evasive** and **occlusion-triggered-near-miss** producers;
- wiring live emission into `build_event_ledger` / the stepping loop (needs per-step
  visibility, track-confidence, and command-source signals verified as exposed).

## Tests

`tests/benchmark/test_safety_predicates.py`: a zig-zag-in-place trajectory is flagged, a
smooth straight run is not, fields/thresholds are correct and overridable, command-source
and velocity sign changes are counted, and invalid inputs (bad `dt`, length mismatch,
single step) fail closed.

## Related

- Canonical safety-event ledger (emit target): `robot_sf/benchmark/event_ledger.py`.
- Real-world trace validation of these labels (externally blocked): #3278.
