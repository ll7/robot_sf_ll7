# PPO development recovery candidate

`dev_delta_recovery_pilot.yaml` runs 983,040 aggregate environment transitions,
24 complete rollouts with 20 environments and 2,048 steps per environment.
It evaluates after each six rollouts. `dev_delta_recovery_15m.yaml` is the full
candidate, pending pilot evidence and domain review; neither config is a release.

The successful variant-B source (`07bd2b8037053f8e979bd97db32d096fe58e5257`)
used signed velocity deltas, including braking without reverse motion. The
development policies trained at `2881f338c90b2e070aa490966f1de8c86d7bc27b`
inherited that declaration but the trainer silently ignored it and the native
environment consumed accelerations. A small command therefore changed speed
by only one tenth as much at the configured timestep, before plant saturation.
The new opt-in environment interface restores the declared contract; default
acceleration environments retain their behavior and config hash.

The candidate restores variant B's rollout length and v3 reward weights, with
one explicit exception: `ttc_risk` is zero because the current reward producer
falls back to the near-miss signal and otherwise charges it twice. It retains
current scenario geometry and pedestrian physics, with a development-only
episode pool and a disjoint development selection pool. No curriculum counter
is involved. The original new policies also used a shorter budget and weaker
progress reward with larger collision/timeout penalties; their individual
causal contributions have not been isolated.

Training now reports completed episode returns, training outcomes, per-term
reward means and PPO approximate KL. These diagnostics were missing from the
failed runs' online histories. Development success is a pilot decision signal,
not held-out performance or promotion evidence. Compare each checkpoint with
the released variant-B final checkpoint under the same native action interface,
scenario manifest, authored horizons and development inputs.

This is a recovery experiment for the legacy differential-drive reference.
It does not train the T60 vehicle planned for 0.1.0. Group behavior changes and
pedestrian calibration require separate evaluation on their eventual source.
Keep frozen artifacts, release scenarios and released checkpoint metadata
unchanged. New-policy checkpoint metadata must declare its actual action
semantics; use acceleration for the previously trained failed checkpoints.

The full candidate uses 15,032,320 aggregate transitions and 983,040-step selection intervals, both whole rollouts. This avoids SB3 rounding every segment upward while the schedule labels a smaller budget. The pilot remains exactly 983,040 transitions.
