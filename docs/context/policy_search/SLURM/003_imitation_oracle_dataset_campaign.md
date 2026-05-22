# Oracle Imitation Dataset Campaign

## Goal

Generate a planner-oracle dataset from the strongest non-training candidate
family and train a faster imitation-style policy that preserves the same safety
guard semantics.

## Gated Dependency

This campaign is blocked on `contracts/oracle_imitation_dataset_split.md`.
Dataset generation, hard-slice augmentation, and relabeling must wait until the
split policy is committed and the manifest schema is populated. See issue #1397.

## Suggested Phases

1. Collect trajectories from the best current model-based candidate.
2. Add hard-slice recovery examples and failure-focused relabeling.
3. Train the policy with config-first reproducibility.
4. Evaluate with the same smoke, nominal, stress, and full-matrix funnel.

## Success Condition

The learned policy is only worth keeping if it matches or improves the current
non-learning reference on the stress slice without increasing collision rate.