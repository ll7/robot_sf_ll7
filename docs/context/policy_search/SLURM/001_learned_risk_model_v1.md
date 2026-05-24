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

Use the pre-SLURM launch packet from issue #1395:

- `configs/training/learned_risk_model_issue_1395_launch_packet.yaml`
- `scripts/validation/validate_learned_risk_launch_packet.py`
- `docs/context/issue_1395_learned_risk_launch_packet.md`

Do not improvise shell-only training commands. A follow-up SLURM issue should first validate the
packet, replace pending durable artifact aliases with concrete trace/baseline/checkpoint artifact
URIs, and record stop gates before training starts.

## Validation Requirement

Run at least `stress_slice` and `full_matrix` after training and compare against
the best non-learning candidate from the current ledger.
