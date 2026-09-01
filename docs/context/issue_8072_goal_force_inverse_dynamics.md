# Issue #8072: observation-only inverse goal force

This implementation estimates a pedestrian's desired force from causal tracked motion while
keeping simulator truth out of the actor-facing belief. It is a bounded shadow-mode diagnostic,
not evidence of calibrated pedestrian-goal prediction.

## Current status

Issue [#8073](https://github.com/ll7/robot_sf_ll7/issues/8073) is already merged and supplies the
public `GoalCandidateGenerationResult` envelope. Issue #8072 now consumes that envelope, accepts
only `GoalForceObservation` rows and explicitly supplied actor-visible force contributions, and
emits the versioned actor-side `GoalBeliefV1` contract.

The canonical implementation is
[`robot_sf/prediction/goal_force_inverse_dynamics.py`](../../robot_sf/prediction/goal_force_inverse_dynamics.py).
It provides:

- H=1 heading-only baseline with no acceleration-derived force claim;
- H=2 finite-difference inversion using the actual elapsed time;
- H=3 causal three-snapshot linear-fit inversion with propagated 2x2 covariance;
- explicit complete, partial, censored, and unavailable information modes;
- actor-side speed-cap inference without consuming oracle cap truth;
- local arrival/braking evidence that cannot turn braking into a reversed goal direction;
- stateful tracking integration keyed by `(tracking_epoch_id, track_id)` with reset handling; and
- a separate evaluator-only oracle-component upper-bound method.

Known force contributions are represented by `ObservableForceComponent`. A zero vector means that
the actor-visible producer checked the component and found no contribution; an unavailable or
omitted component is retained as uncertainty rather than silently treated as zero. Optional
component configuration hashes are carried into diagnostics.

## Reproducible smoke

Run from the repository root:

```bash
scripts/dev/run_worktree_shared_venv.sh -- python \
  scripts/validation/run_goal_force_inverse_smoke.py \
  --config configs/validation/goal_force_inverse_smoke.yaml \
  --output output/validation/goal_force_inverse_smoke.json

scripts/dev/run_worktree_shared_venv.sh -- python \
  scripts/validation/check_goal_force_inverse_smoke.py \
  --input output/validation/goal_force_inverse_smoke.json
```

The smoke reports separate H=1, H=2, H=3, and oracle arms, candidate-provider/config/source
digests, force error, angular error, covariance coverage, component availability, cap/censoring
state, runtime, and the oracle-goal randomization canary. The compact tracked receipt is
[`evidence/issue_8072_goal_force_inverse_smoke_receipt.v1.json`](evidence/issue_8072_goal_force_inverse_smoke_receipt.v1.json).

Observed on the deterministic synthetic fixture: H=2 and H=3 recovered the configured force to
floating-point tolerance, the oracle arm recovered its exact configured component, and the
actor digest was unchanged when an evaluator-only oracle trace was randomized. These are
implementation-integrity smoke observations only; they do not establish force accuracy on
simulator episodes, calibration, human-intention prediction, planner improvement, safety, or a
paper/dissertation result.

## Deferred boundaries

This slice does not add long-horizon hierarchy, waypoint/redirect change detection, forecast
conditioning, learned residuals, policy inputs, or planner-action changes. Those remain gated by
the downstream issues in the canonical #8060 programme. The next evidence step is a simulator-backed
actor-visible force reconstruction comparison after the tracking and force producers are wired
to the same transition-time geometry and parameter provenance.
