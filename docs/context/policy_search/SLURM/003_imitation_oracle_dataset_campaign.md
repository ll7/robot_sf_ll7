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

Issue #1470 now has a concrete Slurm-prepared source-candidate trace job:

```bash
scripts/dev/sbatch_oracle_imitation_traces_issue1470.sh --dry-run --no-status
```

Submit from an allowed Auxme Slurm login node by removing `--dry-run`. The job validates the
launch packet, runs `hybrid_rule_v3_static_margin0_waypoint2` on the packet's exact train split,
and writes a JSONL trace manifest under the job result root. This is executable Slurm work for
the oracle-imitation lane, but it is not yet the final imitation NPZ dataset. Dataset
materialization, hard-slice augmentation, relabeling, concrete durable artifact aliases, and
checksums must still be recorded before imitation training starts.

## Suggested Phases

1. Collect trajectories from the best current model-based candidate.
2. Add hard-slice recovery examples and failure-focused relabeling.
3. Train the policy with config-first reproducibility.
4. Evaluate with the same smoke, nominal, stress, and full-matrix funnel.

## Success Condition

The learned policy is only worth keeping if it matches or improves the current
non-learning reference on the stress slice without increasing collision rate.
