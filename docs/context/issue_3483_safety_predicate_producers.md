# Issue #3483 — Trace-level safety-predicate producers (oscillatory + late-evasive)

**Status:** diagnostic / proxy. **Evidence grade:** idea-level; predicate definitions are
versioned modeling choices, labeled diagnostic until real-world label validation
(externally blocked, #3278). Do **not** report predicate rates in the manuscript until
fields are emitted and validated.

## What this is

`robot_sf/benchmark/safety_predicates.py` provides in-sim producers for the trace-level
safety predicates the thesis motivates, so their fields become reproducible and
fixture-backed instead of simulation-only assumptions.

This increment implements two of the three predicates: **oscillatory local-control** and
**late-evasive reaction**. The oscillatory boolean is compatible with the existing
`SurrogateEvents.oscillation` slot in `robot_sf/benchmark/event_ledger.py`; both detailed
field records are intended for the ledger `surrogate_events` block.

## Producer 1 — `safety_predicate.oscillatory_control.v1`

`oscillatory_control_predicate(positions, headings, linear_velocities, *, dt, …)` emits:

- `heading_rate_sign_changes`, `linear_velocity_sign_changes`, `command_source_changes`
- `net_progress_m`, `path_length_m`, `progress_ratio`
- `mean_abs_jerk`, `n_steps`

Diagnostic classification: `oscillation = (heading_rate_sign_changes ≥ T_h) and
(progress_ratio ≤ T_p)` with defaults `T_h = 4`, `T_p = 0.5`. Thresholds are explicit and
overridable, and the raw fields are always emitted so a different threshold can be applied
downstream without recomputation.

## Producer 2 — `safety_predicate.late_evasive.v1`

`late_evasive_predicate(hazard_distances, hazard_visible, speeds, *, dt, …)` emits
`first_hazard_visible_step`, `conflict_zone_entry_step`,
`first_clearance_restoring_action_step`, `minimum_distance_m`,
`required_deceleration_m_s2`, `response_latency_s`, `n_steps`. The clearance-restoring
action is the first deceleration past `decel_threshold_m_s2` at/after the hazard becomes
visible. Diagnostic classification: `late_evasive` when the hazard is visible and the
reaction is absent, slower than `max_response_latency_s`, or only after conflict-zone entry.

## Scope boundary

Pure and side-effect free — changes no runtime/benchmark behavior. Deferred follow-ups:

- the **occlusion-triggered-near-miss** producer;
- wiring live emission into `build_event_ledger` / the stepping loop (needs per-step
  visibility, track-confidence, and command-source signals verified as exposed).

## Tests

`tests/benchmark/test_safety_predicates.py` (16 tests): oscillatory flagging on a
zig-zag-in-place vs straight run with correct fields/thresholds and sign-change counts;
late-evasive flagging for absent/slow reactions vs a prompt deceleration, latency-threshold
override, no-visible-hazard handling, the required-deceleration formula; and fail-closed
validation for both producers.

## Related

- Canonical safety-event ledger (emit target): `robot_sf/benchmark/event_ledger.py`.
- Real-world trace validation of these labels (externally blocked): #3278.
