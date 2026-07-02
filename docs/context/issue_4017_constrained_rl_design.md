# Issue 4017 Constrained Reinforcement Learning Design

This note defines the first constrained reinforcement learning slice for social navigation:
extract existing RobotEnv safety metadata as costs, apply a Lagrangian reward penalty during
training, and keep claims at design-and-unit-test scope until a later training smoke and diagnostic
comparison run.

## Claim Boundary

- Evidence status: diagnostic design slice only.
- This PR does not train a policy, run a benchmark campaign, submit Slurm or GPU jobs, or make a
  paper-facing safety claim.
- The constrained reward wrapper preserves the raw task reward and attaches constraint diagnostics
  so a later Stable-Baselines3 callback can update multipliers after completed vectorized episodes.
- Fallback or degraded benchmark execution is not used as evidence for this slice.

## Contract

The first supported safety-cost sources are read from `RobotEnv.step()` `info` metadata:

- `collision_any`: top-level collision flag or pedestrian, obstacle, or robot collision metadata.
- `pedestrian_collision`: pedestrian collision metadata.
- `robot_or_obstacle`: robot or obstacle collision metadata.
- `near_miss`: finite non-negative `near_misses` scalar.
- `comfort_exposure`: finite non-negative `comfort_exposure` scalar.
- `ttc_risk`: bounded inverse finite positive `time_to_collision`.

Unknown sources fail closed during constraint-spec construction. Non-finite or negative scalar
metadata is clamped to zero cost rather than propagating invalid values through training rewards.

The wrapper adds these step diagnostics:

- `constraint_costs`
- `constraint_multipliers`
- `raw_task_reward`
- `constrained_reward`

On terminal or truncated steps, the wrapper also adds `constraint_episode` with costs, budgets,
violations, multiplier values before any update, and episode step count. Multipliers are not updated
inside `step()`; later training callbacks must call the explicit update method from completed
episode diagnostics to avoid vectorized-environment update races.
