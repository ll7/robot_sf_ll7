# Oracle Imitation Dataset Campaign

## Goal

Generate a planner-oracle dataset from the strongest non-training candidate
family and train a faster imitation-style policy that preserves the same safety
guard semantics.

## Launch Packet

Issue #1397 adds the pre-Slurm launch packet:

- `configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml`
- `scripts/validation/validate_oracle_imitation_launch_packet.py`
- `docs/context/issue_1397_oracle_imitation_launch_packet.md`

Dataset generation, hard-slice augmentation, and relabeling must still wait for a
dedicated follow-up Slurm issue. That issue should first validate the launch packet,
replace pending durable artifact aliases with concrete W&B aliases or run ids, then
record generated dataset checksums before imitation training starts.

## Suggested Phases

1. Collect trajectories from the best current model-based candidate.
2. Add hard-slice recovery examples and failure-focused relabeling.
3. Train the policy with config-first reproducibility.
4. Evaluate with the same smoke, nominal, stress, and full-matrix funnel.

## Success Condition

The learned policy is only worth keeping if it matches or improves the current
non-learning reference on the stress slice without increasing collision rate.
