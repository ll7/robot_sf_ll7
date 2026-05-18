# Safety Shield Contract

The benchmark safety-shield contract records when a policy action is filtered by a runtime guard.
It is an instrumentation layer, not a formal safety proof.

## Core Types

- `robot_sf.planner.safety_shield.SafetyShield`: protocol for filters that expose
  `choose_command_decision(observation, proposed_command)`.
- `robot_sf.planner.safety_shield.ShieldDecision`: JSON-serializable record containing the
  proposed action, filtered action, decision label, violated constraints, prediction source,
  uncertainty/calibration metadata, fallback-controller state, and selected-action evaluation.
- `shield_metrics_from_stats`: converts per-step shield decisions into benchmark scalar metrics.

## Benchmark Fields

Guarded planners may add these fields under `algorithm_metadata`:

- `safety_shield_contract`: stable description of the shield implementation, prediction source,
  fallback policy, calibration status, and interpretation boundary.
- `shield_stats`: per-episode counters and the last serialized `ShieldDecision`.

Benchmark episode metrics may include:

- `shield_decision_count`
- `shield_intervention_count`
- `shield_override_count`
- `shield_hard_constraint_violation_count`
- `shield_intervention_rate`
- `shield_override_rate`
- `shield_hard_constraint_violation_rate`

These metrics stay separate from reward, success, collision count, and SNQI. A high intervention
rate means the shield changed policy behavior often; it does not by itself prove a safer planner.

## Current Implementation

`GuardedPPOAdapter` is the first shield implementation. It uses a deterministic short-horizon
rollout, not a learned or conformal predictor, and reports `calibration_status=not_calibrated`.
Best-effort decisions such as `fallback_best_effort` or `stop_best_effort` are counted as hard
constraint violations because no evaluated command satisfied the shield constraints.
