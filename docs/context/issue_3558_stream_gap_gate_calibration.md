# Issue #3558 — stream_gap gate-threshold calibration decision layer (increment)

**Status:** diagnostic / proxy. **Evidence grade:** idea-level decision layer; no change to the
production dropping default (a separate decision once a safe region is or isn't found).

## What this is

`robot_sf/planner/stream_gap_gate_calibration.py` is the pure **decision layer** for #3558.
#3471 found that dropping uncertain agents at the *current default* `stream_gap` uncertainty-gate
thresholds increases unsafe commitment. This module turns a gate-threshold sweep — run over the
#3471 episode harness — into the issue's actionable guidance: it classifies each swept setting
against the conservative-retention baseline and reports whether any safe operating region exists,
or confirms that conservative retention dominates.

It mirrors the accepted decision-layer pattern in
`robot_sf/scenario_certification/failure_cause.py` (#3484).

## Decision layer (`stream_gap_gate_calibration.v1`)

- `classify_setting_safety(setting, baseline, tolerance)` → `at_least_as_safe` only when a setting
  worsens no safety axis (unsafe-commit, collision, min-separation) beyond the allowed tolerance.
- `calibrate_stream_gap_gate(settings, baseline, tolerance)` → per-setting classifications, the
  `safe_region`, and either a `recommended_setting` (the safest member: fewest unsafe commits, then
  fewest collisions, then largest separation) with `conclusion = safe_region_exists`, or
  `conclusion = conservative_retention_dominates` when none clears the bar.

## Scope boundary

Pure and side-effect free — changes no behavior. The threshold **sweep** that produces the
per-setting safety aggregates needs benchmark runs over the #3471 harness and is the deliberate
deferred follow-up; this layer turns those results into the gate guidance.

## Tests

`tests/planner/test_stream_gap_gate_calibration.py` (6 tests): no-worse-than-baseline is safe,
worsening any axis is less safe, tolerance allows a small regression, the safest safe-region member
is recommended, retention dominates when none is safe, and empty-sweep rejection.

## Related

- Follows #3471 (PR #3553) — the negative result this converts into gate guidance.
- Sibling ScenarioBelief follow-ups: #3556, #3557.
