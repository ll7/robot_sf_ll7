# Issue #1247 Safety Shield Contract

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1247>

## Goal

Add a lightweight prediction-aware safety shield contract that separates policy actions, shield
interventions, fallback recovery, and hard constraint violations from scalar rewards and headline
benchmark metrics.

## Implementation Boundary

This slice adds benchmark instrumentation only:

- `robot_sf.planner.safety_shield` defines `SafetyShield`, `ShieldDecision`, shield stats, and
  scalar shield metric derivation.
- `GuardedPPOAdapter` now exposes `choose_command_decision(...)` while preserving the legacy
  `choose_command(...) -> (command, label)` API.
- Guarded PPO and guarded GenSafeNav/SoNIC benchmark wrappers record `safety_shield_contract`,
  `shield_stats`, and shield-derived metrics.
- `episode.schema.v1.json` allows the new additive metadata and metrics.

Out of scope:

- no learned conformal predictor,
- no CPO/PPO-Lagrangian/SAC-Lagrangian training,
- no replacement of guarded PPO behavior,
- no formal safety certification claim.

## Validation Plan

The focused proof covers:

- red collection failure before `robot_sf.planner.safety_shield` existed,
- shield decision serialization and stats/rate derivation,
- guarded PPO fallback decisions with proposed/filtered action metadata,
- guarded benchmark wrapper metadata integration.

Full PR readiness should still be run after commit because the schema and map-runner surfaces are
shared benchmark infrastructure.

## Claim Boundary

`shield_intervention_rate`, `shield_override_rate`, and
`shield_hard_constraint_violation_rate` explain how often a shield changed or failed to fully
protect a proposed action. They are diagnostic benchmark instrumentation, not a proof that a planner
is safe in deployment.
