# Auxme Cluster Reference

Cluster: `auxme-imech192`, alias `cluster`.

## Partitions

| Partition | Nodes | Node types | CPUs/node | GPUs/node       |
|-----------|-------|------------|-----------|-----------------|
| a30       | 2     | mix        | 128       | 4× A30 (24 GB)  |
| l40s      | 2     | mix        | 96        | 2× L40S (48 GB) |
| pro6000   | 1     | drain      | 64        | 2× Pro 6000     |

## QoS Profiles and Per-User Job Limits

`QOSMaxJobsPerUserLimit = 2` applies per QoS profile. Each partition has a CPU and a GPU profile:

| QoS profile  | Partition | CPU/node | MEM/node | MAX_JOBS | MAX_TRES_PU |
|--------------|-----------|----------|----------|----------|-------------|
| a30-cpu      | a30       | 40       | 155G     | –        | –           |
| a30-gpu      | a30       | 48       | 250G     | 2        | gres/gpu=2  |
| l40s-cpu     | l40s      | 60       | 188G     | –        | –           |
| l40s-gpu     | l40s      | 64       | 375G     | 2        | gres/gpu=2  |
| pro6000-cpu  | pro6000   | 60       | 562G     | –        | –           |
| pro6000-gpu  | pro6000   | 64       | 1.1T     | 2        | gres/gpu=2  |

**Practical implication:** you can run at most **2 jobs on a30** and **2 jobs on l40s**
simultaneously. To maximise throughput when submitting more than 2 jobs, spread submissions
across both partitions (2 × a30 + 2 × l40s = 4 concurrent jobs). Excess jobs beyond the
per-QoS limit queue rather than run; use `squeue -u $USER` to confirm running vs. pending
state before attributing slowness to job logic.

## num_envs Guidance for Auxme CPU Jobs

For 24-CPU allocations on Auxme, high-throughput PPO runs typically use **14 to 22 envs**.
Do not simply match `num_envs` to the reserved CPU count; leave headroom for the main
training loop and evaluation workers. See
[docs/dev/slurm_resource_audit.md](../../docs/dev/slurm_resource_audit.md) for the full
audit workflow.

## Output File Convention

All scripts use `#SBATCH --output=output/slurm/%j-<description>.out`:

- **Job ID first** (`%j`) so files sort chronologically by default.
- **Rooted under `output/slurm/`** which is gitignored by the root `.gitignore` — no `.out`
  files leak into the repo root.

Example for a new script:

```bash
#SBATCH --output=output/slurm/%j-issue999-my-experiment.out
```

## Scripts in this Directory

| Script               | Purpose                                    |
|----------------------|--------------------------------------------|
| `auxme_gpu.sl`       | Generic GPU job template for Auxme         |
| `auxme_uv_setup.sl`  | Environment bootstrap using uv             |
| `interactive.sh`     | Launch an interactive session on a30/l40s  |
| `issue_791_*.sl`     | Issue-791 training campaign scripts        |

Submit with max wall time via the repo wrapper:

```bash
scripts/dev/sbatch_use_max_time.sh SLURM/Auxme/auxme_gpu.sl
```

## Reliable Issue-791 Submission Workflow

Issue-791 wrappers now require an explicit config path to avoid accidental stage1 fallback.

Recommended submission path:

```bash
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22.yaml \
  --job-name robot-sf-issue791-reward-curriculum \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

The helper does three things before submission:

- prints live partition pressure (`a30`, `l40s`) with free GPUs, pending depth, and per-user slot headroom,
- recommends `partition`/`qos` from the live snapshot when not explicitly passed,
- submits through `scripts/dev/sbatch_use_max_time.sh` so wall-time matches current policy.

Dry-run with recommendation visible:

```bash
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_attention_head_promotion_10m_env22.yaml \
  --dry-run \
  SLURM/Auxme/issue_791_attention_head.sl
```

### Partition guidance

- Prefer spreading long jobs across `a30` and `l40s` to respect `QOSMaxJobsPerUserLimit=2` and maximize concurrency.
- If one partition is saturated (low `free_gpu`, high `pending`, or `slots_left=0`), submit the next job to the other partition.
- Treat `srun: Unable to confirm allocation ... Zero Bytes were transmitted or received` as transient infrastructure noise; resubmit once with identical config.
