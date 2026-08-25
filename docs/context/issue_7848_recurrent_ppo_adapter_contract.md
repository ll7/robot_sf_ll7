# RecurrentPPO Learned Local-Policy Adapter Contract (issue #7848)

## Status

Implementation-smoke owner for the stateful RecurrentPPO local planner
(child C of #7845, after the #7846 frozen contract and #7847 training
runtime). Evidence tier: implementation-smoke — this adapter records no
benchmark, metric, or paper-facing claim.

## What this adapter is

`robot_sf/planner/recurrent_ppo_learned_adapter.py` loads one exact
`sb3_contrib.RecurrentPPO` checkpoint under the #7846 `default_gym`
observation contract (`drive_state` + `rays`), preserves the LSTM
hidden/cell state across control steps, resets it at episode/scenario
boundaries, and emits a desired unicycle command `(v, omega)` in the
canonical planner representation.

It owns:

- checkpoint and policy-class resolution (fail closed);
- observation contract validation (required keys, finite payloads,
  forbidden future/trajectory inputs);
- recurrent hidden/cell state lifecycle (`lstm_states` + `episode_start`);
- reset at construction, explicit `reset(...)`, and episode boundaries;
- deterministic inference mode (default `True`);
- planner lifecycle: `predict` / `plan` / `step`, `reset(*, seed, reason)`,
  `diagnostics()` (versioned, latency, state shape, reset accounting),
  `close()`, `configure(...)`, `bind_env(...)`;
- raw-command observability separate from the clipped adapted command;
- fail-closed handling of missing/corrupt/incompatible checkpoints and
  non-finite observations — no fallback to goal-seeking or another policy.

## What it is not

- No safety wrapper (ORCA/CBF/risk-DWA/emergency stop) and no generic
  wrapper integration — the later execution layer owns that separation.
- No training entry point; no velocity-to-acceleration conversion; the
  canonical `policy_command_to_env_action` path performs conversion during
  environment execution.
- Not registered on any default or release planner roster — the adapter is
  explicitly opt-in by construction.

## Action parity

Raw model output remains the desired `(v, omega)` command, clipped to the
configured bounds for the emitted command while the raw command stays
observable. Downstream `(target-current)/dt` conversion is owned by the
canonical environment action path, not by this adapter.

## Diagnostics

`diagnostics(observation=...)` reports planner/policy identifiers,
recurrent-state shape and finite status, inference call count, reset
count and last reset reason, sequence identity, observation validation
status, raw desired `(v, omega)`, action saturation flags, and bounded
latency statistics (mean/p95/max over the episode window).

## Tests

`tests/planner/test_recurrent_ppo_learned_adapter.py` covers observation
and checkpoint fail-closed cases, recurrent state lifecycle and reset
accounting, determinism default, action clipping, raw-command
observability, planner protocol surfaces, metadata checklist, and the
diagnostics contract with a deterministic stub checkpoint. The planner
suite (`tests/planner`, excluding `slow`) remains green.

## Claim boundary

The adapter is a raw learned planner; it changes no planner default,
safety semantics, benchmark metric, or release roster.