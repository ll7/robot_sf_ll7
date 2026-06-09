# Private Auxme Overlay

Auxme-specific Slurm scripts, node names, partition policy, QoS limits, scratch paths, and
machine notes are intentionally not kept in the public repository.

Use the optional private operations overlay instead:

```bash
export ROBOT_SF_PRIVATE_OPS=/path/to/robot_sf_ll7-private-ops
```

or add the same path to the local, gitignored machine context:

```markdown
- private_ops_repo: /path/to/robot_sf_ll7-private-ops
```

The compatibility wrappers under `scripts/dev/` keep the public command surface stable:

```bash
scripts/dev/auxme_partition_status.sh

scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/<training-config>.yaml \
  --job-name <short-job-name> \
  SLURM/Auxme/<private-script>.sl
```

When the private overlay is configured, those wrappers delegate to:

```text
${ROBOT_SF_PRIVATE_OPS}/auxme/scripts/dev/
${ROBOT_SF_PRIVATE_OPS}/auxme/SLURM/
```

Public, portable experiment contracts should stay in this repository: training configs, artifact
expectations, W&B policy, validation helpers, and compact evidence manifests. Cluster-specific
execution policy should stay in the private overlay.
