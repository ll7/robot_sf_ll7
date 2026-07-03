# Issue #3466 — AMMV command-feasibility / tip-over evaluator + artifact wiring

**Status:** diagnostic / internal-proxy. Thresholds are explicit non-hardware assumptions; no
paper-facing AMMV safety claim before benchmark evidence exists. Artifact-pipeline wiring has
**landed** (evaluator #3600, per-episode + batch wiring #3845, batch claim-boundary parity #3466
follow-up).

> **AMMV** = Autonomous Micromobility Vehicle (three-wheeled tadpole layout). See
> [`glossary.md`](../glossary.md) if unfamiliar. Everything here is an *internal proxy*: it flags
> commands a three-wheeled platform likely could not execute, **not** a hardware-calibrated claim.

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

`evaluate_artifact_command_feasibility(commands)` adapts the pure evaluator to the map-runner
artifact surface (per-step mappings with `linear_velocity` / `angular_velocity`) and fails closed
(`status: missing_inputs`, `feasible: false`, `tip_over_violation: true`) when commands are missing,
malformed, or holonomic-only — the three-wheeled proxy cannot infer yaw feasibility from `vx/vy`.

## Artifact wiring (where the fields appear)

- **Per-episode** (`robot_sf/benchmark/map_runner_episode.py`): `run_map_episode` records the
  evaluator output under `algorithm_metadata.ammv_feasibility` for every episode
  (`algo_meta["ammv_feasibility"] = evaluate_artifact_command_feasibility(ammv_command_actions)`).
  Additive under `algorithm_metadata`, so existing episode rows stay backward-compatible.
- **Batch summary** (`robot_sf/benchmark/map_runner_batch_summary.py`):
  `accumulate_batch_metadata` folds per-episode payloads into `feasibility_totals`;
  `build_ammv_feasibility_summary(feasibility_totals)` emits the aggregate
  `algorithm_metadata_contract.ammv_feasibility` block.

### Claim boundary carried on both surfaces

Both the per-episode payload and the batch-summary block carry the same claim-boundary markers so a
consumer reading **only** the summary cannot mistake the aggregate for calibrated hardware evidence:

- `evidence_kind: "diagnostic_proxy"`
- `proxy_kind: "internal_non_hardware"`
- `status: "available"` (or `"no_ammv_episodes"` at batch level when nothing contributed, so a
  `None` margin with `feasible: false` is interpretable rather than silently ambiguous).

Batch folds are **worst-case**, never means: `min_stability_margin` is the minimum across episodes,
`tip_over_violation` is OR, and `feasible` requires every contributing episode to be feasible.

## Tests

`tests/benchmark/test_ammv_feasibility.py`:

- Evaluator: feasible sequence passes, over-yaw trips tip-over, excess curvature flagged, in-place
  yaw limit, agreement with `metrics.evaluate_stability_margin`, length/empty/param validation,
  non-finite fail-closed.
- Artifact adapter: versioned-field extraction, fail-closed for missing yaw-rate / invalid inputs.
- Batch fold: `test_batch_metadata_folds_ammv_feasibility_fields` (totals), plus
  `test_batch_summary_block_carries_claim_boundary_and_worst_case_fields` and
  `test_batch_summary_block_fails_closed_without_ammv_episodes` (summary block: claim boundary +
  worst-case reduction + no-episode fail-closed).

`tests/benchmark/test_map_runner_utils.py` asserts the per-episode record exposes the versioned
`ammv_feasibility` block with the four acceptance fields.

## Related

- Tip-over computation source of truth + #3479/#3587 reconciliation: `robot_sf/benchmark/metrics.py`,
  `robot_sf/robot/rollover_proxy.py`.
- Runtime rollover proxy (in-sim, complementary surface): #3479.
