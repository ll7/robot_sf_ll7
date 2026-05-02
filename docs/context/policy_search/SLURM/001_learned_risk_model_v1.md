# Learned Risk Model v1

## Goal

Train a lightweight risk estimator that predicts collision, near-miss, and low-
progress risk from local rollout features. The model is not safety critical on
its own; it is only allowed as an auxiliary cost term alongside a hard guard.

## Inputs

- stress-slice and full-matrix episode traces for implemented non-learning candidates,
- candidate command features and local crowd geometry snapshots,
- benchmark outputs rooted under `output/`.

## Expected Deliverables

- training config under `configs/training/` or `configs/policy_search/` if new code is added,
- trained checkpoint stored under the canonical model registry flow,
- evaluation report comparing guarded model-based planner with and without the learned risk term,
- update to `candidate_registry.yaml` and `experiment_ledger.md`.

## Suggested Command Shape

Use a future config-first entrypoint. Do not improvise shell-only training commands.

## Validation Requirement

Run at least `stress_slice` and `full_matrix` after training and compare against
the best non-learning candidate from the current ledger.