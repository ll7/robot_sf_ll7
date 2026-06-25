# Issue #3466 — AMMV command-feasibility / tip-over evaluator (increment)

**Status:** diagnostic / internal-proxy. Thresholds are explicit non-hardware assumptions; no
paper-facing AMMV safety claim before benchmark evidence exists.

## What this is

`robot_sf/benchmark/ammv_feasibility.py` evaluates whether a planner's command sequence respects a
three-wheeled AMMV's tip-over and proxy non-holonomic limits — flagging commands that look fine under
the current robot abstraction but a three-wheeled platform could not safely execute. It does **not**
change planner behavior.

It reuses the benchmark-surface source of truth `metrics.evaluate_stability_margin` (reconciled in
#3587) for the tip-over margin, so this lens and the `rollover_min_stability_margin` column stay
consistent (a test cross-checks this).

## Evaluator (`ammv_feasibility.v1`)

`evaluate_command_feasibility(velocities, turn_rates, params)` emits `min_stability_margin`,
`tip_over_violation` + `n_tip_over_steps` (+ `rollover_event` = `ROLLOVER_CRITICAL`),
`n_curvature_violations` (proxy non-holonomic `|ω| ≤ max_curvature·v`, with an in-place yaw allowance
at near-zero speed), and an overall `feasible` verdict. `AmmvFeasibilityParams` defaults align with
the benchmark-surface stability geometry.

## Scope boundary

Pure and side-effect free — no planner-behavior change. Surfacing these fields in benchmark
artifacts / evidence summaries (the artifact-pipeline wiring, Tier-2) is the deliberate deferred
follow-up.

## Tests

`tests/benchmark/test_ammv_feasibility.py` (7 tests): feasible sequence passes, over-yaw trips
tip-over, excess curvature flagged, in-place yaw limit, agreement with
`metrics.evaluate_stability_margin`, and length/empty/param validation.

## Related

- Tip-over computation source of truth + #3479/#3587 reconciliation: `robot_sf/benchmark/metrics.py`,
  `robot_sf/robot/rollover_proxy.py`.
