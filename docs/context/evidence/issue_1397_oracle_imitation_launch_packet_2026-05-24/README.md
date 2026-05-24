# Issue #1397 Oracle Imitation Launch Packet Evidence

This directory contains the small tracked dry-run fixture used by the #1397 launch-packet
validator. It is not a trajectory dataset and must not be used for imitation training.

Files:

- `dry_run_dataset_stub.json`: minimal fixture referenced by
  `configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml`
  so checksum coverage and artifact-path validation can run before Slurm collection.

Checksum:

```text
eb5ef938d15725ff29a013a196216d093dd549ae58ae805e642c98441777529f  dry_run_dataset_stub.json
```
